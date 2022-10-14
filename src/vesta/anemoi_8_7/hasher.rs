//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use super::{Jive, Sponge};

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// An Anemoi hash instantiation
pub struct AnemoiHash {
    state: [Felt; STATE_WIDTH],
    idx: usize,
}

impl Default for AnemoiHash {
    fn default() -> Self {
        Self {
            state: [Felt::zero(); STATE_WIDTH],
            idx: 0,
        }
    }
}

impl Sponge<Felt> for AnemoiHash {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 32];
        for chunk in bytes.chunks(31) {
            if num_hashed + i < num_elements - 1 {
                buf[..31].copy_from_slice(chunk);
            } else {
                // The last chunk may be smaller than the others, which requires a special handling.
                // In this case, we also append a byte set to 1 to the end of the string, padding the
                // sequence in a way that adding additional trailing zeros will yield a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 32];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 31 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[i] += Felt::read(&buf[..]).unwrap();
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
                num_hashed += RATE_WIDTH;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        let sigma = if elems.len() % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        let mut i = 0;
        for &element in elems.iter() {
            state[i] += element;
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // 2*DIGEST_SIZE < RATE_SIZE so we can safely store
        // the digests into the rate registers at once
        state[0..DIGEST_SIZE].copy_from_slice(digests[0].as_elements());
        state[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(digests[0].as_elements());

        // Apply internal Anemoi permutation
        apply_permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiHash {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        assert!(k % 2 == 0);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = vec![Felt::zero(); STATE_WIDTH / k];
        let c = result.len();
        for (i, r) in result.iter_mut().enumerate() {
            for j in 0..k {
                *r += elems[i + c * j] + state[i + c * j];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0xde867569f86ecf44,
                0x1875bc7465e36071,
                0x57c15230bb667207,
                0x13ec47056654e2f1,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xf68bf15bacd5924c,
                    0x6a357419cf399d17,
                    0xf53e558e4d5541a1,
                    0x01d52700b86b6e70,
                ])),
                Felt::new(BigInteger256([
                    0x89be5ed2c0c26a7a,
                    0x1b0474d4c46254e1,
                    0xa9def18c0a0d9d68,
                    0x1543074ddbe10e7e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x09df04bceebbee77,
                    0x810d361aa02a091f,
                    0xf913f2bb058732de,
                    0x2d305ccb78c6eafd,
                ])),
                Felt::new(BigInteger256([
                    0x76bf049a216e8427,
                    0x1edb39c7e43941a0,
                    0x1445a9540d195dc9,
                    0x3372c8d044033e24,
                ])),
                Felt::new(BigInteger256([
                    0x43ee86bc483f1c0f,
                    0x38347342ae211eb7,
                    0xb37bde4e7ff2847c,
                    0x3ea93de5ffa83ad8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3df2f98eb817b4c7,
                    0x9f0004c86ebf48ac,
                    0x650c85756c9c32da,
                    0x1c2009193b7140cc,
                ])),
                Felt::new(BigInteger256([
                    0x8ae4c7bd1808f8a7,
                    0x294232c290e5c678,
                    0x6842645e5164f9fa,
                    0x01c05084396ec2ae,
                ])),
                Felt::new(BigInteger256([
                    0xa4a6df2107ef599a,
                    0x46b0b2b117c1995a,
                    0x9027c5e6ef243108,
                    0x0eb66c71ab3c46ba,
                ])),
                Felt::new(BigInteger256([
                    0x29eb8606ac232f47,
                    0xc2a47a1bd22325df,
                    0x221c60d756a0bd2c,
                    0x16a36d7399d34b4e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0c5c556b80a76848,
                    0x7e8e0aed35425140,
                    0x26dc9b611a6c93ff,
                    0x0de8139094b7218b,
                ])),
                Felt::new(BigInteger256([
                    0x351efec6e11602c8,
                    0x3749772a35ece6f7,
                    0x88214dc465b72e72,
                    0x33d7afc14dd06170,
                ])),
                Felt::new(BigInteger256([
                    0x012d10f21b99aa95,
                    0xa174a4a6c68a6c56,
                    0x6ff43c31f04521d0,
                    0x0bf1685df3dd00d4,
                ])),
                Felt::new(BigInteger256([
                    0x035c0d593375a512,
                    0x7d907b9ca2c0cd97,
                    0x790f7eb8aa45f12f,
                    0x2d6996b27de71559,
                ])),
                Felt::new(BigInteger256([
                    0xf8183918ae329476,
                    0x6fff8f72190a9bfa,
                    0x0d7f0562ad1661e8,
                    0x2ba5db7b8226a6ea,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbe982e1bcca64c10,
                    0xe8e6094dff0f8c9c,
                    0xd78eae875e7287b4,
                    0x1114b4b172ff3fdc,
                ])),
                Felt::new(BigInteger256([
                    0xf91ae4b9a70108d1,
                    0x80f299d8e4649212,
                    0x92afcef38445f42d,
                    0x28c074e3593b29df,
                ])),
                Felt::new(BigInteger256([
                    0x052da0727f10b73a,
                    0xefc7f28e3eda9334,
                    0x60469ad0143c06c5,
                    0x3410f6b60def43a5,
                ])),
                Felt::new(BigInteger256([
                    0xd74fcd2b2ad3f7d2,
                    0xbc2dacd4f8d9b35f,
                    0xb128dedac8373db0,
                    0x126e200b1bf44d5d,
                ])),
                Felt::new(BigInteger256([
                    0xb8d7c49149f41fb1,
                    0xe6d24c9ba30eaee2,
                    0x2190896a23e2e354,
                    0x3041da0eb65c0a8f,
                ])),
                Felt::new(BigInteger256([
                    0x603b1de7d8e44102,
                    0x54af94b58413b186,
                    0x62cb733b3fb12000,
                    0x1d0c60e8b8d1052b,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x891c36f1813c93dc,
                0x459d8a9c3df37bea,
                0x6c65c32abe913346,
                0x14f1a6c0cdcd487b,
            ]))],
            [Felt::new(BigInteger256([
                0xe1b64de716dd426d,
                0x394720bb5213520e,
                0xce19d46994fdaf3e,
                0x168cce8270befaac,
            ]))],
            [Felt::new(BigInteger256([
                0xd70f42eb892ce4ac,
                0x3fd3d63937245fd5,
                0x81848699ebaf0714,
                0x1d90a7af8edf1738,
            ]))],
            [Felt::new(BigInteger256([
                0x368133e132074e54,
                0x22f35dcedcda4e86,
                0x13fa8a2c31d7fc74,
                0x27ab418cdcfb4047,
            ]))],
            [Felt::new(BigInteger256([
                0xa43ebe2ca91f6122,
                0x15938c4ff5a2d734,
                0x0aca55e5f6447c94,
                0x246281e9606cff1a,
            ]))],
            [Felt::new(BigInteger256([
                0xaa365d6da40430b6,
                0x70050a639826f526,
                0x15d4f41bdbb857f0,
                0x05064730acadc0c9,
            ]))],
            [Felt::new(BigInteger256([
                0x7c9f7ce89c53ff9e,
                0x36da6ba6e7e454c7,
                0x63124ccc06a5ef0a,
                0x0f724e76ee7c9ff0,
            ]))],
            [Felt::new(BigInteger256([
                0x513178f66b2018bf,
                0x90cac13c7af4816d,
                0xad84c4aaaf126ff9,
                0x12f0572e399c568c,
            ]))],
            [Felt::new(BigInteger256([
                0xb762bcff1ab72fb1,
                0x0c4582e5b2f4b3d4,
                0x372d5c94388d964b,
                0x31d62ae8e5f14c83,
            ]))],
            [Felt::new(BigInteger256([
                0xa0ecf28d735356e3,
                0xc81b85e84e553db4,
                0x997fbf23304969e9,
                0x1effed641c8a662d,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0af8044175a7b633,
                    0x2fc4db5eb37d869c,
                    0x41f30103c8a25a98,
                    0x379d0b38658a573b,
                ])),
                Felt::new(BigInteger256([
                    0x6dcbf84ca7c153ea,
                    0xdc5adf4099e7b53c,
                    0xb1ce4c54d11a8996,
                    0x173d83be50e2c6b8,
                ])),
                Felt::new(BigInteger256([
                    0xac941e903b120e5f,
                    0xb0b73bc21a9adb90,
                    0xe47d2cf765b61dd6,
                    0x1acc1e172be1bb70,
                ])),
                Felt::new(BigInteger256([
                    0xb70399a93360101d,
                    0xbc7a0d424312a483,
                    0x3eea56a2e3a03de3,
                    0x390b7a8b0471b928,
                ])),
                Felt::new(BigInteger256([
                    0xc1b3b0134c634a79,
                    0x8b12e8a98c94fd57,
                    0x2b1436cc3914add7,
                    0x2faa61e47b45e5d1,
                ])),
                Felt::new(BigInteger256([
                    0xeadb1b6d1118e2b4,
                    0x249a864cdb394a2f,
                    0xa7511894f91b1ca3,
                    0x2a02eb69a675dd7a,
                ])),
                Felt::new(BigInteger256([
                    0xda7fd39a54b51fd7,
                    0x7dad7d966dccb9c9,
                    0xda3c877a58bbe88d,
                    0x3eb87556053c9a83,
                ])),
                Felt::new(BigInteger256([
                    0x3d1b0c68e317eb52,
                    0xd1ff6e9585a7a6a8,
                    0x53d4b05cfd3caf29,
                    0x2da6539e81b4fe75,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x834e7ad5a79861a3,
                    0x7a8ac52bd321fd24,
                    0x5d979c2ca5839427,
                    0x13ef86851d2198ab,
                ])),
                Felt::new(BigInteger256([
                    0xfea1238fa2476cea,
                    0x6ef15155592bd733,
                    0xf87ac6c08a683e71,
                    0x3dc0603999e699f0,
                ])),
                Felt::new(BigInteger256([
                    0x4a9dae8c607e37d6,
                    0xfeb9c4cbf83f3f95,
                    0xdbbf582326339043,
                    0x1e38300ee0ca2e27,
                ])),
                Felt::new(BigInteger256([
                    0x0068df75af7592d8,
                    0x0b593fe421a5a6bb,
                    0x56c07c9c77f37b59,
                    0x23acfab001b31595,
                ])),
                Felt::new(BigInteger256([
                    0x36155b8ac0cfe633,
                    0xda9ab523891474eb,
                    0xf6ed1b3f52671bb9,
                    0x22068d77dd1c11b1,
                ])),
                Felt::new(BigInteger256([
                    0x7f5fd6f485a895ca,
                    0x5a82ba92575a71fe,
                    0xd03130bd4aea948b,
                    0x03ff08d134b5f714,
                ])),
                Felt::new(BigInteger256([
                    0xa6f38d13a3ba1dc9,
                    0x4e772d85043b6ce0,
                    0x73de31f9d42f55e2,
                    0x0ee133a05b4b18d1,
                ])),
                Felt::new(BigInteger256([
                    0x406e061a6b57ef8f,
                    0x616da13319129d07,
                    0x97ba99d3a6a11898,
                    0x3ef7016e22ddf602,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf29d7f4de6b4701d,
                    0x8a51d1ba05854386,
                    0x63aaf2f2cf9c66f5,
                    0x3511733d69caec03,
                ])),
                Felt::new(BigInteger256([
                    0x0449431c15e73ef8,
                    0xf0312ea714c5df75,
                    0x3093ff17a4a0c541,
                    0x0ba3d9d8bc34f3af,
                ])),
                Felt::new(BigInteger256([
                    0x52979d09ea4a4bc3,
                    0x488d53158c7b3046,
                    0x6272e13a017ca003,
                    0x175f169c94a88181,
                ])),
                Felt::new(BigInteger256([
                    0xe647c42b141aa8bb,
                    0xfa1a7b4c04ccd193,
                    0xeac7953fb4a39c8b,
                    0x39b882ce19c94234,
                ])),
                Felt::new(BigInteger256([
                    0x4cc98fed0ef452ac,
                    0x9271d7ae666a8444,
                    0xdd4f48cf686bb761,
                    0x22f245d049cc90ee,
                ])),
                Felt::new(BigInteger256([
                    0xb667d6aee93a512f,
                    0x5bdb0274386c167b,
                    0x4204ffa9b300a694,
                    0x2361e77164374ee4,
                ])),
                Felt::new(BigInteger256([
                    0xcb57cbafc864e0c8,
                    0x6f2e5355609fca07,
                    0x00745a82e552e22f,
                    0x230430088d38ed19,
                ])),
                Felt::new(BigInteger256([
                    0x2e4417f7a5a196a0,
                    0x6877327f327f394c,
                    0x894a3e6ef2384bc3,
                    0x2d184e10fd069c88,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x61eac32e21b84b00,
                    0xd5da06b756b56dd9,
                    0x29a3d1a25e5689c7,
                    0x01771c0b615db3cb,
                ])),
                Felt::new(BigInteger256([
                    0x272717655794606e,
                    0xca7799b7a29bbef1,
                    0x607adae6a3b58f2b,
                    0x24b0711b55246d20,
                ])),
                Felt::new(BigInteger256([
                    0x9f1201e3f450b99b,
                    0x985882b6d7adea17,
                    0xd593aa4103ecc4cd,
                    0x1b9014802195f1b2,
                ])),
                Felt::new(BigInteger256([
                    0xced447fb74cff105,
                    0x94ae439b3ff464c5,
                    0xa6232706305f9eb4,
                    0x00109659db1bff2a,
                ])),
                Felt::new(BigInteger256([
                    0x0b2ca31e756fde11,
                    0x6f2a0083907f2d73,
                    0xcce2d1bd0cde6294,
                    0x0aae05be326ab335,
                ])),
                Felt::new(BigInteger256([
                    0x36ce859ac8fffef3,
                    0xce7b01f03a2e392e,
                    0x5455dd0a86a3c637,
                    0x3712b991e1f4833d,
                ])),
                Felt::new(BigInteger256([
                    0x2ed09cfde2491691,
                    0x5f61b26082f7fc5b,
                    0xd8d249f41ecbc029,
                    0x09630a9cffc4b054,
                ])),
                Felt::new(BigInteger256([
                    0x9970f3086af0ef23,
                    0xe1d24eae60bf04e1,
                    0x64201532e521be30,
                    0x09cc3d54518e4a48,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa0c17df227b497f1,
                    0xb3da06eb2a7eec24,
                    0x8cda1799e7de0417,
                    0x3bcfea99f6f2f396,
                ])),
                Felt::new(BigInteger256([
                    0x9ae9ef72a154bb1b,
                    0x6276378508f7a303,
                    0xd4f5e720a446b5fd,
                    0x263776811bc0b878,
                ])),
                Felt::new(BigInteger256([
                    0x424b6b6ec40eedc2,
                    0xfe65eb9580cd9145,
                    0xfef79c0fa69227a2,
                    0x209e60c65a1c9562,
                ])),
                Felt::new(BigInteger256([
                    0x8a77ae23462871b8,
                    0x9d4ba3c907609b22,
                    0xd27156405e81bc22,
                    0x0ea58527a1e896c6,
                ])),
                Felt::new(BigInteger256([
                    0x5d083c0734c2d11d,
                    0x772ce938affa12e6,
                    0x82431a7c5c794692,
                    0x2de22179c2740fb9,
                ])),
                Felt::new(BigInteger256([
                    0xb43870ee03f933e3,
                    0x09ea99a80a644df5,
                    0x58f1fd43fdf5a04d,
                    0x00072c7e3937edbe,
                ])),
                Felt::new(BigInteger256([
                    0xaf3fe087984a87aa,
                    0x4fb4cbe8919e67cb,
                    0x42f45e214f5cfe46,
                    0x37bd4be159ada991,
                ])),
                Felt::new(BigInteger256([
                    0x17fad01fcd8a3aac,
                    0x2513fdf0e83243d7,
                    0x35400069178d240c,
                    0x2f23fd50a5936ff1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc99393cfd26d5b2c,
                    0x0b2813d9ffb741b9,
                    0xbc1e187010f0e793,
                    0x0af6e448e9844709,
                ])),
                Felt::new(BigInteger256([
                    0x6697880342014f90,
                    0xf0019c5a58e04f9f,
                    0x38a4b78200086bc8,
                    0x305b924a92c0458c,
                ])),
                Felt::new(BigInteger256([
                    0xb205af6320419aeb,
                    0x3e5e25cb322d1447,
                    0x9c474470a1c1ddac,
                    0x312cbb3633c2d277,
                ])),
                Felt::new(BigInteger256([
                    0xeab5ffbdd93e900f,
                    0x64f231a8911ea834,
                    0x35d49d702669c2fc,
                    0x3eb4203d7ba05868,
                ])),
                Felt::new(BigInteger256([
                    0x986ef2f44e84218a,
                    0xe5789d6e2b1ffcf8,
                    0x0fe83c4e6e42b4b7,
                    0x060820f0732bc8d9,
                ])),
                Felt::new(BigInteger256([
                    0x0aa72ec861b7f3f2,
                    0xbe3d557ac434c82b,
                    0xd69ee24a2b758a01,
                    0x2a4597f0293beca9,
                ])),
                Felt::new(BigInteger256([
                    0xf3e8aaaca02797f2,
                    0x9e0a8261ca7943ee,
                    0x1932f0cc9f54dd55,
                    0x033e9133815feba0,
                ])),
                Felt::new(BigInteger256([
                    0xa0b9faab12c7a1b0,
                    0xe93d20a1c68a54af,
                    0x57cb7ecb4e5da4db,
                    0x283a75365a56c080,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xc001e07677908573,
                    0xf32b428aab604994,
                    0x067de7d4c0db4cbc,
                    0x04fb5dc508e7f4f7,
                ])),
                Felt::new(BigInteger256([
                    0x5461b68d5b560a16,
                    0x2db191684499249f,
                    0xf03a148ea4e37db4,
                    0x3f8a698a2d7d5a04,
                ])),
                Felt::new(BigInteger256([
                    0xbadcec93e60a5280,
                    0x43d66d082e049f6a,
                    0xfdc49ffa932c0721,
                    0x06f78696f109d091,
                ])),
                Felt::new(BigInteger256([
                    0xd3448de90d11b929,
                    0xff8163dc0cd7df5d,
                    0x7623ca712b5d0f2c,
                    0x064fe0bd67f7b9bc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9936e0a91d4019d9,
                    0x19c11f70f2176368,
                    0x27977f8a6a80fb50,
                    0x0542df2f3dbd2ce6,
                ])),
                Felt::new(BigInteger256([
                    0xae0e3d2898859eb7,
                    0x39abcc94c22f3a6c,
                    0xf5aae890ae8b98be,
                    0x27ef3f026acce19c,
                ])),
                Felt::new(BigInteger256([
                    0x398546480391061d,
                    0x018f90feb265d79a,
                    0x2f84de44f5a87dbf,
                    0x12bff4fa3c9c87ed,
                ])),
                Felt::new(BigInteger256([
                    0xfcbe4ea0387aae66,
                    0x6adb02084f10aa50,
                    0x32649de55df54780,
                    0x35fda793e670f723,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x410fdac8e40e5e63,
                    0x96980e641ec0ebaa,
                    0x49bce0e7415e29da,
                    0x10e8ff4adcabdd62,
                ])),
                Felt::new(BigInteger256([
                    0xc62f5173d82eef35,
                    0x89b78c85b077458c,
                    0x97ef6a2e95c0c747,
                    0x299452c8935555f3,
                ])),
                Felt::new(BigInteger256([
                    0x77df9b69ea006727,
                    0xbb4d519d2bf2b42e,
                    0x1ab335de955e738e,
                    0x29b75441d980dab4,
                ])),
                Felt::new(BigInteger256([
                    0x0934e76026306257,
                    0xf8d428efb9197969,
                    0xaf54ab50ebf584e8,
                    0x18e8386e48c794c0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x422613d9ca3e3eb0,
                    0x2a8b44ddcb41b27e,
                    0xbba23c35d428fb45,
                    0x16529ea0cfc0cf42,
                ])),
                Felt::new(BigInteger256([
                    0x0e610ab601c8f352,
                    0x0387600bbee377ea,
                    0x315fdead88a1ac84,
                    0x1bd295f9c85820a0,
                ])),
                Felt::new(BigInteger256([
                    0x0df276173d2dd5d6,
                    0xc2ed3a54692d5685,
                    0x3c5d2f9a69f17f0a,
                    0x10923a22f968ca4d,
                ])),
                Felt::new(BigInteger256([
                    0xb776ec476409b703,
                    0xd7e09d8fc69a7491,
                    0x01fe1b986fe281b8,
                    0x009c8436957361e1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2fe2b34156cca11a,
                    0xeb0789f5c3321e23,
                    0xc2a680232ef4d3c1,
                    0x29cd56ab1ff78bf3,
                ])),
                Felt::new(BigInteger256([
                    0x3808ce963b7f3b5c,
                    0x40884260b61a6282,
                    0x788e8fc9c9b45b93,
                    0x17e6777b8198a167,
                ])),
                Felt::new(BigInteger256([
                    0xa828ef28de73b275,
                    0xe0735869877a60ac,
                    0x4c5095a9a716a163,
                    0x09b34dbd5d78d86e,
                ])),
                Felt::new(BigInteger256([
                    0xdac3901a054f1a41,
                    0xc7f98b532165d4c9,
                    0x335ea5bc1c37e7ac,
                    0x36b525ae01e7013d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x583df4334a61f331,
                    0x637710562a6dcac3,
                    0x21f268755630d0fd,
                    0x2be821dd28b70802,
                ])),
                Felt::new(BigInteger256([
                    0x66ca593426e4293f,
                    0x00ce8d9e29224022,
                    0xab6b24cb3b4eccda,
                    0x2b2e4eb74a8682fd,
                ])),
                Felt::new(BigInteger256([
                    0x497721a97497d35b,
                    0x16b298394cfdda10,
                    0x33bd458a0515f4e9,
                    0x219ae75be1db689b,
                ])),
                Felt::new(BigInteger256([
                    0xe174bcba6cc2b688,
                    0x7ac47db062257af7,
                    0x2d0b1654972eecfc,
                    0x20a1ee29cb809dcf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc2c81cfe9ba54cd9,
                    0x675559b56069d110,
                    0x8e1d22d87941cc94,
                    0x33a56b095d04035f,
                ])),
                Felt::new(BigInteger256([
                    0x6f9dc9af076cc895,
                    0x97b5b9431ca882bb,
                    0xe023cedbe5c079ce,
                    0x125edebbd5f2856b,
                ])),
                Felt::new(BigInteger256([
                    0x295818c14568a5be,
                    0xa5a94ef10a21e133,
                    0xe2e320aa8fa3ef17,
                    0x1c3220b57b49fb9d,
                ])),
                Felt::new(BigInteger256([
                    0xee1af1ebda8f1810,
                    0x7887f0a4e19fb878,
                    0x8f54417ed6043c34,
                    0x243677aadc1cf0d4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x682beafe37dbdd3e,
                    0x087cbd54dc557790,
                    0x206412a4b43689c6,
                    0x307d9cd5831f9d4e,
                ])),
                Felt::new(BigInteger256([
                    0xaf82433db8c2680a,
                    0xc345a2c4abb3a35b,
                    0x0ae9a13831fb9cf7,
                    0x0d44214c2ed9f6a4,
                ])),
                Felt::new(BigInteger256([
                    0xfd5c233725321a00,
                    0xdbca7ab5a1ce52a0,
                    0x135dff35d5340ead,
                    0x30065c8529820499,
                ])),
                Felt::new(BigInteger256([
                    0x97816c69caa5c847,
                    0x650453a2cb42a576,
                    0x4d2538925bb1ddb9,
                    0x31c2441480e952b1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x09d3d1d9a45a06c9,
                    0x176b89ec991c38f5,
                    0x29208986da4b7ff2,
                    0x019475f425891845,
                ])),
                Felt::new(BigInteger256([
                    0x756e15391ced2a0d,
                    0x3e78538e537d864c,
                    0x5e8083cb544e13a3,
                    0x054cff5ece80a354,
                ])),
                Felt::new(BigInteger256([
                    0x8b7f43e199d8fbf6,
                    0x80adc8463351f8f7,
                    0x97349cafb3793816,
                    0x0b4fd8e1a890ad44,
                ])),
                Felt::new(BigInteger256([
                    0x2018b9406480710b,
                    0x523ae3584cef4aa3,
                    0x3e87d07447d5e348,
                    0x32d518bbbf4fd6da,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x37be26743505f3a8,
                    0xa430e69024a23104,
                    0xd3db4332c6684e9c,
                    0x3d68a163d3e024ee,
                ])),
                Felt::new(BigInteger256([
                    0x33c01bba84861bd6,
                    0x2a4895d3dc14837b,
                    0xb45f20fc5d9fbc85,
                    0x24a1815cd08e99bd,
                ])),
                Felt::new(BigInteger256([
                    0xda1f4697f45a1da9,
                    0x7e662464ed15569d,
                    0x6659e76f03997b30,
                    0x3c5c8dd5bd68623f,
                ])),
                Felt::new(BigInteger256([
                    0x39345eb1fcff6df7,
                    0xa9b0bc8d3cd0c7cf,
                    0x080e11b15d055aa7,
                    0x398bb427ee28ec29,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0ab11a6c1da814d1,
                    0x854ae3f1decfbdec,
                    0x61446d4ecb3e347c,
                    0x05666d27a3e670c9,
                ])),
                Felt::new(BigInteger256([
                    0x179a2d4e54edde17,
                    0x04fbb8ef8baf4d42,
                    0x8d10d5cfb8c6ef21,
                    0x2ea9fd997050e0d1,
                ])),
                Felt::new(BigInteger256([
                    0xa413ad16b639c7df,
                    0xa1f09fa6adf23d9e,
                    0x2feebfdf6c7387b8,
                    0x25b2e7932dd5aa93,
                ])),
                Felt::new(BigInteger256([
                    0xcb89d99ec6085a31,
                    0xd6916f66baca8a7a,
                    0x66ff0d4ccd3478a0,
                    0x28262873a8307487,
                ])),
                Felt::new(BigInteger256([
                    0x90579ea3a3b9ba1d,
                    0xb0c04c30de85adde,
                    0x47d0c47170d9b284,
                    0x1416489c98c810c4,
                ])),
                Felt::new(BigInteger256([
                    0x68ccf4029371655f,
                    0xad723bc9ebc98543,
                    0x51a5b2fc28469d1a,
                    0x393b9dd89f237e65,
                ])),
                Felt::new(BigInteger256([
                    0x6ec33d025ea6163f,
                    0xe57b394b2564e351,
                    0x511058688ca3e14e,
                    0x3b9609eb323f2411,
                ])),
                Felt::new(BigInteger256([
                    0x1b40f8fc6baca142,
                    0xeabacaa51454ed5f,
                    0x8401a45382dd2b50,
                    0x3f3fed52b08afef8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd6d0e8ab8e1fc6e3,
                    0x97bd0fd3adee2c40,
                    0x497ac015abc3070d,
                    0x2e12cce7e1e97923,
                ])),
                Felt::new(BigInteger256([
                    0x8b63e8de8c8bca25,
                    0xb1509554c275e8cb,
                    0x807306028b66465d,
                    0x26c5fabc966eb5b7,
                ])),
                Felt::new(BigInteger256([
                    0x0e97ad7d42e6748e,
                    0x5a401eb8eda84f12,
                    0xf9e16d3e247dc748,
                    0x315adc91bcae65d2,
                ])),
                Felt::new(BigInteger256([
                    0x00f01fb050f7d540,
                    0x3033d37f40be65cb,
                    0x739c01844b3e4428,
                    0x3cd1b6be989a32bc,
                ])),
                Felt::new(BigInteger256([
                    0xd0b9227dd3401c06,
                    0xee45457ff7a5166a,
                    0x391aefaece3f8f86,
                    0x0d5ee911bc639c8a,
                ])),
                Felt::new(BigInteger256([
                    0x46cc2b3787268f60,
                    0xbc5277241ea4825c,
                    0x2d7fbbbd63fb5bfc,
                    0x0327a7ef9784ad53,
                ])),
                Felt::new(BigInteger256([
                    0x77873b4d9f9d4d23,
                    0x7db67fadd8cce8e3,
                    0x1fe31c5b3bb168c5,
                    0x33af774689149b63,
                ])),
                Felt::new(BigInteger256([
                    0x95bd799e82ab1e68,
                    0x3f41c54920743a34,
                    0xdd82fef734b73b04,
                    0x21b749b31802fb6f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaa0f8aac07799228,
                    0x0d72f8933d163d90,
                    0xb5de413326c63432,
                    0x08ccd851a593ef0b,
                ])),
                Felt::new(BigInteger256([
                    0xe14ea69574e7aef3,
                    0xf84eded2b8e15808,
                    0x5a7a29cf5ac31d03,
                    0x385017b4e883a7ae,
                ])),
                Felt::new(BigInteger256([
                    0x85578936798048ef,
                    0x0890b3e2a35349aa,
                    0x24ec9097ac273f58,
                    0x234aa88cfa097930,
                ])),
                Felt::new(BigInteger256([
                    0xe27b806ee04dbc44,
                    0xcd1672c8dd583aac,
                    0xcb90c0e79beef40f,
                    0x3cd8159c70dbd25c,
                ])),
                Felt::new(BigInteger256([
                    0xa0e23b81f3b75dce,
                    0xcab1e6b43f568105,
                    0x5d91d24583022003,
                    0x3b4f57ada20af0e6,
                ])),
                Felt::new(BigInteger256([
                    0x8569b3399493258f,
                    0x3cba9fa41b51f446,
                    0xe8517bdba13fc161,
                    0x08dfc555caeba156,
                ])),
                Felt::new(BigInteger256([
                    0x1ba6e18b7d5d5c84,
                    0x5512f6af1e529f92,
                    0x4638a7ef618b221c,
                    0x3f5bc06cfce98916,
                ])),
                Felt::new(BigInteger256([
                    0xbfa53950c9c78550,
                    0xef91d7ac980f8850,
                    0xd275040be3ab5f9b,
                    0x15e08083533a6e03,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2bb3fd90d38d5fa6,
                    0x2ddb2e1f3368c0ba,
                    0x6e70900312666cdb,
                    0x39c2f56b1d19da30,
                ])),
                Felt::new(BigInteger256([
                    0x1d1bd22c4eba638e,
                    0x08013248b7356e8d,
                    0xf517ab4590b20533,
                    0x368251a22711506f,
                ])),
                Felt::new(BigInteger256([
                    0x59c3b867de3f0a5c,
                    0x414acab544bc9faf,
                    0xc93069e3ff92b8ef,
                    0x008d52d7a15ece82,
                ])),
                Felt::new(BigInteger256([
                    0x75d23053aed74002,
                    0xb806a6634c620f2e,
                    0xd1d1f8d2b7bbd8a5,
                    0x1cbdfc8d6d32517e,
                ])),
                Felt::new(BigInteger256([
                    0x65a1342611b43ed8,
                    0xede255e29dcb6573,
                    0xe1c230186f9eecad,
                    0x1ad27794e80e1731,
                ])),
                Felt::new(BigInteger256([
                    0xb9a1097734e37c88,
                    0x112335e1b3ec4851,
                    0x22d81b7cd203eee8,
                    0x152d7e32f7bbc73d,
                ])),
                Felt::new(BigInteger256([
                    0x9a9fc35facbcabe9,
                    0x6133d6d0aa256e4f,
                    0x66fe093723c53c92,
                    0x108a2126c04f4fab,
                ])),
                Felt::new(BigInteger256([
                    0x063eec0fdcad81e8,
                    0xf944deee53c3cb6d,
                    0x8a26abc64c1d367d,
                    0x1796610c47ef3ade,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2e69cab64bda482a,
                    0x9fbdbe711866427a,
                    0x2a62b1b74255f49d,
                    0x05d48a896c53dfd5,
                ])),
                Felt::new(BigInteger256([
                    0x29efd1a38e6b8d1c,
                    0x9c9e6641ef41c328,
                    0x09ce6b6db59c10ec,
                    0x2961aacffa04ae03,
                ])),
                Felt::new(BigInteger256([
                    0x86311781a42269dc,
                    0x29d5b02f63842945,
                    0x25ca06ad78da557d,
                    0x2d40e0025de93194,
                ])),
                Felt::new(BigInteger256([
                    0x5bea754acfecbde8,
                    0xa798a0bad8a6b1c4,
                    0x6317b8fad19ba11e,
                    0x198217634beff3b6,
                ])),
                Felt::new(BigInteger256([
                    0x1ef5063ddfb0eaad,
                    0x71cd1758fd21d3da,
                    0xb6bedf4058b7ddbe,
                    0x271e6b52852698cf,
                ])),
                Felt::new(BigInteger256([
                    0x20e4443a59190db2,
                    0x598d8076f3b540fb,
                    0xbb8a7822c68ff3b1,
                    0x18073a5d28c69d15,
                ])),
                Felt::new(BigInteger256([
                    0x8698ae009963665c,
                    0xee59c67ce966dcae,
                    0x3fee54a1ac5496c6,
                    0x05b7724fe18a5b96,
                ])),
                Felt::new(BigInteger256([
                    0xddc1e83af10f8fc0,
                    0xfa9671731aa45fed,
                    0x037f9712a4cfe226,
                    0x0a94e837ae142842,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x15dedc4cbe03ce6d,
                    0xcb99faf31a70c40f,
                    0x0e2810b881fbce34,
                    0x3baeae23bbca019a,
                ])),
                Felt::new(BigInteger256([
                    0x7ed5371280353ac4,
                    0x5aadc9f268327170,
                    0x4aa799020919686c,
                    0x1697c9d5d771d454,
                ])),
                Felt::new(BigInteger256([
                    0x249f64a90c99c742,
                    0x5346785ed92d5e88,
                    0x04863602d1d52ccd,
                    0x34ca49ec4e2f9027,
                ])),
                Felt::new(BigInteger256([
                    0x326dcb3c8301e655,
                    0x38619c074f13cc7c,
                    0x7ee2294c20f3e5ec,
                    0x237898fb96eb8a86,
                ])),
                Felt::new(BigInteger256([
                    0x3422b4ca24545b7e,
                    0xbf9451a20e094d13,
                    0xa83a6629683474d6,
                    0x2b5167eb49d3729e,
                ])),
                Felt::new(BigInteger256([
                    0x763b65b9fe5e1c97,
                    0xf5775fe94ac15a0c,
                    0x1747c49e1ecc1f36,
                    0x35d72911ca7e7c97,
                ])),
                Felt::new(BigInteger256([
                    0xf9bd1e9a5f3da89a,
                    0xfa519c41fe8db5b9,
                    0x78fe84f0cebddb22,
                    0x0f86f3382a809ca9,
                ])),
                Felt::new(BigInteger256([
                    0xb892d2100c5bf055,
                    0xa6c80966c8bfb1de,
                    0x21e0682c739a2c94,
                    0x194e3222bf5cc824,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x163e265fc6029b31,
                0x41ee0bdb2141441f,
                0x6aa066cf2447e0bf,
                0x11cd2ea38f66d94a,
            ]))],
            [Felt::new(BigInteger256([
                0xf141c798f1d16d12,
                0x9d90e610ac2876e2,
                0x7f2be4456caa594d,
                0x35efbabfcb978d93,
            ]))],
            [Felt::new(BigInteger256([
                0xfc0cc3e5cc6e1715,
                0xb22a7c7aaaafb5f0,
                0xabb42c455872e999,
                0x3d1cdec39249a2ca,
            ]))],
            [Felt::new(BigInteger256([
                0x89a995cd6d3ebeda,
                0xa699e3d1b0584ca1,
                0x2b5d6616369ea88c,
                0x0353f2f426f51c11,
            ]))],
            [Felt::new(BigInteger256([
                0xfe66996905b11a74,
                0x0d31f8f694600c1b,
                0xb79e4336aca3cc9f,
                0x375a031e7e9b0db6,
            ]))],
            [Felt::new(BigInteger256([
                0x9106b611ce6d416f,
                0x2f7009206407a80a,
                0xe213a8d475c381ce,
                0x2c3caf51ab8c2bda,
            ]))],
            [Felt::new(BigInteger256([
                0x2a96bcb861844396,
                0xea7e144ac762752f,
                0x49abdeb0ff8ee44e,
                0x3b7f39dc2787dba6,
            ]))],
            [Felt::new(BigInteger256([
                0x3d684c597aa9e4bd,
                0x3164238b40b259e2,
                0x255d65998c687790,
                0x22ba847ca7f08287,
            ]))],
            [Felt::new(BigInteger256([
                0x89d75fc0796b282b,
                0xeb109c850dc390a4,
                0xf5b4ef07e7b9c4b0,
                0x2105130cd077f323,
            ]))],
            [Felt::new(BigInteger256([
                0x85f636fc50ffe70b,
                0x2a33656cf0f8fd85,
                0x18274efff1c86b9b,
                0x31414aa348d8955d,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
