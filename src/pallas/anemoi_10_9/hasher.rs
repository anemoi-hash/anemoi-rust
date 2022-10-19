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
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x75aa5b7a4ea9b5e5,
                0xa14e34a6abb8bc48,
                0xc62ec5799a1fb184,
                0x148ce7432ec96e70,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x514acb67dd70e3e2,
                    0xdf74805aa34ecc41,
                    0xe3d8bc2e95f46935,
                    0x05bd8e58e5ffd745,
                ])),
                Felt::new(BigInteger256([
                    0xe2ed360f0964d618,
                    0xe7b7dc699b213a08,
                    0xf09c60fd72348dba,
                    0x185b9f3ec37c9c96,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xddc8d21d8ecc81f1,
                    0x56299ade80e4f90f,
                    0x1178cc0e6c1e8e16,
                    0x2d141b1f8c5568db,
                ])),
                Felt::new(BigInteger256([
                    0x03525bd9542ccef2,
                    0xf51e114d914a2836,
                    0x96c290bbfe9f2869,
                    0x1f7a6f888990a28d,
                ])),
                Felt::new(BigInteger256([
                    0xd11c39ba1c91563b,
                    0x5c7dbc965749d6f8,
                    0x60606a1439fb89ed,
                    0x0eac913006fd4f5e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2f038285d6cd78a5,
                    0x1e5ac4a294e5d8bc,
                    0xf9137391d07aa343,
                    0x2e0e363c50e37355,
                ])),
                Felt::new(BigInteger256([
                    0xc9d405ee0ace99af,
                    0x139f355f78e8c4a9,
                    0x645a7998d0bed4b0,
                    0x38a85c63d409e2ba,
                ])),
                Felt::new(BigInteger256([
                    0xd07af14cb7ae7fb4,
                    0x2dd2f1507abfe206,
                    0x637c8d596ee6202a,
                    0x10be922e1e5f030b,
                ])),
                Felt::new(BigInteger256([
                    0x48c83b956318c629,
                    0xd5c1e31ec67de2b4,
                    0x6964cc64c772c1cf,
                    0x063470dbbcf1265a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x565802dbff25e9fa,
                    0x94413c5c6d20b16c,
                    0x346dabec08830c51,
                    0x2366e679732ea706,
                ])),
                Felt::new(BigInteger256([
                    0x20a5b69589a967fd,
                    0x6dceaf0df6f783d2,
                    0x14f1815ef66dc211,
                    0x31261d3ed68a0de7,
                ])),
                Felt::new(BigInteger256([
                    0x8da40102ae6711a0,
                    0x60dd077ca86df1d0,
                    0x01b3b9f6433bc318,
                    0x2cb243eb0249d3cb,
                ])),
                Felt::new(BigInteger256([
                    0x06235b22d09f3a01,
                    0x7766ec7449c3be3c,
                    0x83913ce6dbc2e4ac,
                    0x2af983ddfbaba005,
                ])),
                Felt::new(BigInteger256([
                    0x7eaeefaf354b4690,
                    0xf1fa1a74156049a7,
                    0xb339bf92c3adf048,
                    0x2f1efe281522ba60,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x967ef819eb7a510b,
                    0x51c0b3a92b176fc1,
                    0x32c4f79c567b5bf6,
                    0x16df3e45781855da,
                ])),
                Felt::new(BigInteger256([
                    0x5308094277c84df0,
                    0xf90a7cdee05cdafa,
                    0xc48fbcfa6db4290d,
                    0x1220195e6ac5c65b,
                ])),
                Felt::new(BigInteger256([
                    0xb93c1ff642dda9eb,
                    0x52372adf61333670,
                    0x0a7f159e7ce558fd,
                    0x13d8c1f6df1b7f93,
                ])),
                Felt::new(BigInteger256([
                    0x81cee3d67f0db591,
                    0x658421f4760dbfa9,
                    0x713b70e49e662108,
                    0x2da14116c36365e4,
                ])),
                Felt::new(BigInteger256([
                    0xc7fa2aa31f8e477d,
                    0xbb7d21e3ad574ff9,
                    0x959dc3b0ad218fe1,
                    0x0ceff3ae62d560cc,
                ])),
                Felt::new(BigInteger256([
                    0x2a5ce33113252da9,
                    0xfa6120615817a44e,
                    0x9f2ea7634d0f7dfa,
                    0x3d475166c961ad30,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x9e4260b1186bd444,
                0x62b050c59d5b5e9b,
                0x4bc9ee515e9a0186,
                0x38d8d818ed64c2cd,
            ]))],
            [Felt::new(BigInteger256([
                0x4056ff2f8dcaec37,
                0x01e19ab77e656e9f,
                0xe6e4174003f88318,
                0x38b58606f2ba2306,
            ]))],
            [Felt::new(BigInteger256([
                0x2e279ddbdfad8bf8,
                0xa596b2d93d18f537,
                0xb144f4224604c2d7,
                0x19e92aba9b160eb8,
            ]))],
            [Felt::new(BigInteger256([
                0x101ec1741eeb3200,
                0x9c2806418429a40a,
                0xef2c63d02c455292,
                0x28d0b4083729988d,
            ]))],
            [Felt::new(BigInteger256([
                0xb4c19829eefc61ae,
                0xb6001c3afb2ffac9,
                0x3a62979d9c0b293f,
                0x3dd94a8e04a0bfad,
            ]))],
            [Felt::new(BigInteger256([
                0x73e7ec6b4e6f4909,
                0x1618472ca691a8da,
                0x3c0e45adb26b90a1,
                0x3f4e406850667bc9,
            ]))],
            [Felt::new(BigInteger256([
                0xf8aff67e62980263,
                0x7c9daee444b14a85,
                0x8b5d7ff083b861c2,
                0x0c034db6e9569b36,
            ]))],
            [Felt::new(BigInteger256([
                0xdb64a1ab50ac9f2a,
                0x9df9426036f03a1f,
                0xefa7ba05fc45945d,
                0x2b3b3fb921a07eeb,
            ]))],
            [Felt::new(BigInteger256([
                0x150c37a210e1af09,
                0x5384ff69f8837ca1,
                0xe9a592a777b8272f,
                0x3cae8aad631abc1e,
            ]))],
            [Felt::new(BigInteger256([
                0x988b76bd54ce2ab1,
                0x642916db3296bcb6,
                0x4430601d2a48fc2a,
                0x2323aeeb87f4deca,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x9e4260b1186bd444,
                0x62b050c59d5b5e9b,
                0x4bc9ee515e9a0186,
                0x38d8d818ed64c2cd,
            ]))],
            [Felt::new(BigInteger256([
                0x4056ff2f8dcaec37,
                0x01e19ab77e656e9f,
                0xe6e4174003f88318,
                0x38b58606f2ba2306,
            ]))],
            [Felt::new(BigInteger256([
                0x2e279ddbdfad8bf8,
                0xa596b2d93d18f537,
                0xb144f4224604c2d7,
                0x19e92aba9b160eb8,
            ]))],
            [Felt::new(BigInteger256([
                0x101ec1741eeb3200,
                0x9c2806418429a40a,
                0xef2c63d02c455292,
                0x28d0b4083729988d,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 310];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);
            bytes[124..155].copy_from_slice(&to_bytes!(input[4]).unwrap()[0..31]);
            bytes[155..186].copy_from_slice(&to_bytes!(input[5]).unwrap()[0..31]);
            bytes[186..217].copy_from_slice(&to_bytes!(input[6]).unwrap()[0..31]);
            bytes[217..248].copy_from_slice(&to_bytes!(input[7]).unwrap()[0..31]);
            bytes[248..279].copy_from_slice(&to_bytes!(input[8]).unwrap()[0..31]);
            bytes[279..310].copy_from_slice(&to_bytes!(input[9]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb8f9fcb6df374572,
                    0xc53c8b3e83052abc,
                    0x05c4cc7f8b40a8ad,
                    0x3cb361191ba7919f,
                ])),
                Felt::new(BigInteger256([
                    0x020d02b5d68cdb14,
                    0x9c2227511d0377e5,
                    0x286e7ebde06740bd,
                    0x3b04ddb73e24ee7f,
                ])),
                Felt::new(BigInteger256([
                    0x23ac3d233828284f,
                    0x9a40c3dfe54bce5c,
                    0xdfd4f83fb7ca3bcc,
                    0x2fb26dfc462588df,
                ])),
                Felt::new(BigInteger256([
                    0x59d5c596725e5cf1,
                    0xf885c615a17a9195,
                    0x54e63bdde0582461,
                    0x018335d48af1fb3a,
                ])),
                Felt::new(BigInteger256([
                    0x1a29f1a47e2ee6e2,
                    0x2bb10a89a05ac681,
                    0xac67c1edc6197fb3,
                    0x1ebd7ba181d822d2,
                ])),
                Felt::new(BigInteger256([
                    0x6f2debfd7ab6e72d,
                    0x04fff7b5907f5dcf,
                    0xb2efe23f60e2ec61,
                    0x3eda3aaef880de08,
                ])),
                Felt::new(BigInteger256([
                    0x009c910fc462a346,
                    0x6576e6a7f4afb314,
                    0x657a24b727a3bef3,
                    0x1a421f4030ee924b,
                ])),
                Felt::new(BigInteger256([
                    0xb39ae5f313f1c080,
                    0x1e9613ba93a9ee95,
                    0xd628291ede6d2aa3,
                    0x2a4d8097bebe7d35,
                ])),
                Felt::new(BigInteger256([
                    0x1e8738f6961e17fc,
                    0x8b0b26af181fd966,
                    0xde8ab59eda6b11a2,
                    0x3ba7b632b75898ff,
                ])),
                Felt::new(BigInteger256([
                    0x0949727fad746822,
                    0xbd1f220fad659f54,
                    0xc9ce7bdf9a7bb8ea,
                    0x2f8c68cc1f6c1fbc,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x56849c504d7d55e3,
                    0xcb123a5654969a5f,
                    0x3ebe8e2e90b614ec,
                    0x3b75093872b8ae1d,
                ])),
                Felt::new(BigInteger256([
                    0x6f800bbcc543687e,
                    0x683f87cc665a239b,
                    0xc14d3f53f7478fa8,
                    0x14cf9731ecb51256,
                ])),
                Felt::new(BigInteger256([
                    0x2803e9c2873a1e03,
                    0x7f117f65084bdc04,
                    0x0b2a7c0f1969fda4,
                    0x316c9b22825b1f5c,
                ])),
                Felt::new(BigInteger256([
                    0x16a8bd415f6ab27e,
                    0xa708809e3c4b75ac,
                    0xf1ce11ba59adb7ba,
                    0x252ddd1483f9d381,
                ])),
                Felt::new(BigInteger256([
                    0xb5fb7b9bdf295612,
                    0x4edb878ad2302f45,
                    0xc06ffe87373e1083,
                    0x3383a60ef74403cd,
                ])),
                Felt::new(BigInteger256([
                    0x664f41d2c55d4d2c,
                    0x58ee89f66f39df0c,
                    0x558a8ee7cc943082,
                    0x07c45aa8b26c2f13,
                ])),
                Felt::new(BigInteger256([
                    0x38055d9f7bb34397,
                    0x7f4ba59f3dd2e8a7,
                    0x7ee79c7bb8b8b38c,
                    0x139f3f021fe4c95e,
                ])),
                Felt::new(BigInteger256([
                    0xba10091b34b6cc0c,
                    0x98d94b402bc611f9,
                    0x3538a3ac1cc2e602,
                    0x082670e5b47acc6c,
                ])),
                Felt::new(BigInteger256([
                    0x96e090e24a4ec1d2,
                    0x38cc6c8157ca484f,
                    0x2ce63b40bb539970,
                    0x32270eac72d9a3c9,
                ])),
                Felt::new(BigInteger256([
                    0x44b447bc6d864f17,
                    0xf3f08f24a9b5cbec,
                    0xda32f197efea07cd,
                    0x2f1b4d28d2edea9f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x576231933aa8bbdf,
                    0x196ed8e62953bca0,
                    0x10f7208fb757b4fb,
                    0x33163439de334ab8,
                ])),
                Felt::new(BigInteger256([
                    0x214e380225e33808,
                    0xf2fd7771baf8ec76,
                    0x20340de54a3f8693,
                    0x25af3f829ce44611,
                ])),
                Felt::new(BigInteger256([
                    0xb1e32a2aa9895990,
                    0x4c89fd28ad43bf54,
                    0x11284e03c14f2f0f,
                    0x22f70388ceaa044c,
                ])),
                Felt::new(BigInteger256([
                    0x0db2eef31c5a3812,
                    0x40e125ec14bad8d8,
                    0x32221342151b6106,
                    0x3aa6a2001c0e5fc2,
                ])),
                Felt::new(BigInteger256([
                    0x32bad3d129496821,
                    0x4ffbefdc5a328fb6,
                    0x936c1c08129c1311,
                    0x1fdb1241f19d3262,
                ])),
                Felt::new(BigInteger256([
                    0x5c0011c18e40076c,
                    0xc075625064217a00,
                    0x986d64f5d4351316,
                    0x3b2e5e6e527a2624,
                ])),
                Felt::new(BigInteger256([
                    0x363b6cdbf0dfd24c,
                    0xff35508841fd36f7,
                    0x9e70db7885821578,
                    0x37863cb841467ebf,
                ])),
                Felt::new(BigInteger256([
                    0x8f38f994d635b6fa,
                    0xa8504c98fbbba8d0,
                    0x09dbaef94224784e,
                    0x3ec5c9cea3265574,
                ])),
                Felt::new(BigInteger256([
                    0x31c4613ccdb67abb,
                    0x42007bef89256c1a,
                    0xa45f65796bdb95aa,
                    0x340aceab44efb0af,
                ])),
                Felt::new(BigInteger256([
                    0x3e7965657fe247df,
                    0x921594d9579a4818,
                    0x0f0d1b8c9fac46d4,
                    0x33d609754ff94f9b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x10599316e2e905ea,
                    0x4da7d10a1c453cfb,
                    0xfee3733ca1893ddc,
                    0x0e3eee48941fa2da,
                ])),
                Felt::new(BigInteger256([
                    0xed6f7538b67b1b99,
                    0xf0fa31015c638e45,
                    0x0a3e6334611729ce,
                    0x190b63eceeecef02,
                ])),
                Felt::new(BigInteger256([
                    0x1991741fb892095c,
                    0xa15af60429616ff6,
                    0x10818cd7e94cb7ab,
                    0x20ee12d0497f4efd,
                ])),
                Felt::new(BigInteger256([
                    0x32231e007c8aee01,
                    0x4883391d932dcffc,
                    0x7b428ac12b1828e3,
                    0x289bb8b6f3ba4f8f,
                ])),
                Felt::new(BigInteger256([
                    0xc5647de3af20124e,
                    0x8dcae0e7beb6b3ca,
                    0x570eae814ccad7e0,
                    0x0bf578f0c85b2910,
                ])),
                Felt::new(BigInteger256([
                    0x09fb14e3b417e361,
                    0xe149fe683d4bdd73,
                    0xc8533a5e6e30da92,
                    0x3ae0e874081efdf2,
                ])),
                Felt::new(BigInteger256([
                    0x14eb3367f56b6a44,
                    0xea3fb5c125d7cb51,
                    0x4e61a28106f6deaa,
                    0x02db350a24ee7cc9,
                ])),
                Felt::new(BigInteger256([
                    0x1e057ce66c078a23,
                    0xc53a5aef53a027c7,
                    0x4a09e76e29b85612,
                    0x0263874b445aea09,
                ])),
                Felt::new(BigInteger256([
                    0xfa628e5f48760673,
                    0x2277139d161bf697,
                    0xdfb2bc9c0c290fab,
                    0x084abfd6b2f8b4bf,
                ])),
                Felt::new(BigInteger256([
                    0x4aa7ea60de2e1c90,
                    0x89684b82da9c9afc,
                    0x9268242048965a51,
                    0x13818e01adcff6c7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2efcad0bca463633,
                    0xc176193afe5bb2ef,
                    0xd8df32d751fd44e2,
                    0x35c02dffa1fbb4eb,
                ])),
                Felt::new(BigInteger256([
                    0xe85557693324c2b2,
                    0xc071525515ba7544,
                    0xc39596518ba6aecb,
                    0x0d15aa2d5ac587c2,
                ])),
                Felt::new(BigInteger256([
                    0xdeb41b3f085bf370,
                    0x18040d2d707c3ad8,
                    0x1f864dbfd8c2a068,
                    0x22d210e6263018c5,
                ])),
                Felt::new(BigInteger256([
                    0x95abaa0a0fe0dd2f,
                    0x0a3d0244e4dad18f,
                    0x20ffd8d8bbd61222,
                    0x0358d08f84854df9,
                ])),
                Felt::new(BigInteger256([
                    0x462d2a2e9ed53576,
                    0x9cac4ed7c2a437b0,
                    0x9d00dcd813a7faec,
                    0x1192e8dfdbc63858,
                ])),
                Felt::new(BigInteger256([
                    0x9f809f39ed827185,
                    0xb0cf1760f54edca4,
                    0x79614a2842904e3d,
                    0x126de8ebfa2d0d76,
                ])),
                Felt::new(BigInteger256([
                    0xffe22174679189bb,
                    0x53037b00a95a773f,
                    0x787a12d813e8be9b,
                    0x3398d08f1b3c7240,
                ])),
                Felt::new(BigInteger256([
                    0x693f49dcb784998c,
                    0xa36b354225003d63,
                    0x8544e0e04decb221,
                    0x06096e61f8874d77,
                ])),
                Felt::new(BigInteger256([
                    0x8d1c18ecf6d779e7,
                    0xc04d63e059c97440,
                    0xd14d268d6c86a31c,
                    0x16e2dcae5fc2632b,
                ])),
                Felt::new(BigInteger256([
                    0xeba5830fdebd65e5,
                    0x721938465fadbff5,
                    0x79d6d922f3380226,
                    0x37332f631a4f407b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x29562c6300861f41,
                    0x63307ca40208a514,
                    0xc26bb5edb0591147,
                    0x1729a54a0c8eb6e1,
                ])),
                Felt::new(BigInteger256([
                    0x335fdc518d047622,
                    0xa24f8c3d8213f675,
                    0xeaa717ca0afc0260,
                    0x21ba00c2b6dadfd6,
                ])),
                Felt::new(BigInteger256([
                    0x31c247f0193d1ed7,
                    0x073543be8b658263,
                    0x81176d1f36a58ba7,
                    0x1599c5de9a2665e2,
                ])),
                Felt::new(BigInteger256([
                    0xb6fae456d1d843e2,
                    0x9aa6eac73d7db558,
                    0x35c5dd0462e52125,
                    0x3c6050e4d883aa93,
                ])),
                Felt::new(BigInteger256([
                    0xe08525cdecf2b31a,
                    0x0a2099105231ce1b,
                    0xf010f4031d8b0c0e,
                    0x326b9cc31e60b173,
                ])),
                Felt::new(BigInteger256([
                    0xed8f470287cde5e3,
                    0x8f87dbc461f5b039,
                    0x4490a3bc4b9d70d8,
                    0x18e78169389baf45,
                ])),
                Felt::new(BigInteger256([
                    0x238db4de43430978,
                    0xd627b35b4a3d8280,
                    0xf4d9ebb1198b35ea,
                    0x3df945e33c41cbdf,
                ])),
                Felt::new(BigInteger256([
                    0x270433aef5c56815,
                    0x6e64fc11c2095b58,
                    0x2402fec64ded0da7,
                    0x2c71eb3eecfe9d35,
                ])),
                Felt::new(BigInteger256([
                    0xc6b3ee2cdd56d217,
                    0x0f68d5587f163173,
                    0x8887338edfb1385e,
                    0x160d9199bc4941db,
                ])),
                Felt::new(BigInteger256([
                    0x773cf29830fd401a,
                    0x209d7d98c1a30a9e,
                    0x797cda28ab327421,
                    0x1c42a3699a24ef2c,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xbe8dca9b6909eca6,
                    0x5e3d8c3ac1008ed8,
                    0x4040203794db40bb,
                    0x0638e607faa6899b,
                ])),
                Felt::new(BigInteger256([
                    0x382158f76d916644,
                    0x946862681a4216c4,
                    0x498b0f5ae334e70a,
                    0x0a880aa0fbce1bf4,
                ])),
                Felt::new(BigInteger256([
                    0x5e38a54251e2c439,
                    0xd4a7b83cd3619b37,
                    0xd1fcf96583144aca,
                    0x076e540282c9b88a,
                ])),
                Felt::new(BigInteger256([
                    0xa18f19d0749f91ab,
                    0xb2cbaf16dee0de1e,
                    0x49250186276f0e37,
                    0x0420f5b333e2a229,
                ])),
                Felt::new(BigInteger256([
                    0xd401a2b2a5277470,
                    0xe911bc81f605edb5,
                    0x7dbbc021c79ab97a,
                    0x3e483892eb21fa64,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc76e18413d73d145,
                    0xb03c59a06d13e4f8,
                    0xb834966bb6228e08,
                    0x16ea79e2326973b1,
                ])),
                Felt::new(BigInteger256([
                    0x93cb940905c34740,
                    0x24082569c9877b25,
                    0x109c69840c0bc2cd,
                    0x08ded87dfbb8f970,
                ])),
                Felt::new(BigInteger256([
                    0x96eba2ccccf8f6b5,
                    0x39cc919bc1d04569,
                    0x7151a63aff1cd96f,
                    0x36cfce0f30cde1d9,
                ])),
                Felt::new(BigInteger256([
                    0x5fce2ec13a99b894,
                    0x38b0aeaf4c62379a,
                    0x95bd2ae67ddac606,
                    0x24c99a5dd73d04f1,
                ])),
                Felt::new(BigInteger256([
                    0xda41632c528f6df2,
                    0x1982936efd58f56a,
                    0xcfebd8bd2c511780,
                    0x2540016d992b4325,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3543b0ad85cc8015,
                    0x38b687e5af833cc6,
                    0x54f9a8ad4ab8fea1,
                    0x3c6c558cc066509c,
                ])),
                Felt::new(BigInteger256([
                    0xb9a85693e2559b69,
                    0x616b2f79135c2392,
                    0x1970857641644fbc,
                    0x296ba234b309ba74,
                ])),
                Felt::new(BigInteger256([
                    0x3ffec1a63d2bda2b,
                    0x0b6381c2d44d9e7f,
                    0x42e4470a36176401,
                    0x1b7ec769e9e79f1f,
                ])),
                Felt::new(BigInteger256([
                    0xbd7f84d279e470d3,
                    0x8aab449505d4e898,
                    0xbad5df895ede6eb5,
                    0x2afb704a8460cc8a,
                ])),
                Felt::new(BigInteger256([
                    0x829a57e05c546817,
                    0xa646da1ad588f1e3,
                    0xf26796eb1c3a2114,
                    0x25ce050270bd181d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xddc5420a1f6a05c2,
                    0x9a3fc8abb69548dc,
                    0xd9b8c8b79a84a7ca,
                    0x30610f0766afef6e,
                ])),
                Felt::new(BigInteger256([
                    0xec4ba47836fef071,
                    0x5cf178bf6af71289,
                    0xd58032094eb69a67,
                    0x335f82c2aeaab162,
                ])),
                Felt::new(BigInteger256([
                    0x51b1eeeae07d0c0d,
                    0x15b9f20817991a54,
                    0xe0217a20488fd029,
                    0x3e332d0feaf8a74e,
                ])),
                Felt::new(BigInteger256([
                    0xb87b3db3187d3e4e,
                    0x16ed76e1ac969bb7,
                    0x54a61c59183be4b3,
                    0x064360487b1ccbea,
                ])),
                Felt::new(BigInteger256([
                    0xa215223862533096,
                    0x90953e2006ae30c8,
                    0x6df544a451c1ee6b,
                    0x394ce39063863d75,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcf3ebe1d9d6093b2,
                    0x6225660fd55069c0,
                    0x1c8db09ba04694e7,
                    0x26049660c9d4de36,
                ])),
                Felt::new(BigInteger256([
                    0x75ac7f7152e5135a,
                    0x63a8b4821bcf62b6,
                    0xfc6db149daeaae3c,
                    0x2655a8a35eb01a63,
                ])),
                Felt::new(BigInteger256([
                    0x2d020a7626a03d65,
                    0x90657c8e9cfaf183,
                    0x98e9e1a62674d35a,
                    0x173a66ef11995405,
                ])),
                Felt::new(BigInteger256([
                    0x4b5e8fe4ea0fb878,
                    0x26f2af08f56bed3c,
                    0x771ebc8b1eec76df,
                    0x2b5e59803b468b36,
                ])),
                Felt::new(BigInteger256([
                    0x245f3146fc93b9e7,
                    0x922bc76ac9d0057f,
                    0xe78158433a414e6f,
                    0x1caa8460752958c6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9618b00bb7a7e4c9,
                    0x748615eeb4060c36,
                    0x74e989a9593a8cca,
                    0x13d21e6b809aff34,
                ])),
                Felt::new(BigInteger256([
                    0x13e109dc96b2389a,
                    0xd80c7ac6dc94189d,
                    0xd55e8b48dbebeedd,
                    0x09e98f1fd85f2953,
                ])),
                Felt::new(BigInteger256([
                    0x648c1a8d16dc1436,
                    0x31463482b44faad9,
                    0x9bae267967c6d8bc,
                    0x0e66b8b13a541261,
                ])),
                Felt::new(BigInteger256([
                    0xd3f499a4c5dd33a0,
                    0xaa7bc5be2a808201,
                    0xea11b31d97be44af,
                    0x2f28fd24e179a363,
                ])),
                Felt::new(BigInteger256([
                    0x1a84ff86762a7261,
                    0xa3e6535f89cbd126,
                    0x479a0e3add7dad6e,
                    0x0e511514039b1aac,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3ac7b8e7e5e233cc,
                    0x31976621bf099ae4,
                    0xacaf2a94be6b54d6,
                    0x1aa061c12d3aefdf,
                ])),
                Felt::new(BigInteger256([
                    0xf4e3a153f4962a3a,
                    0xfcaf88fd559f4dc8,
                    0xd75a446cfb61a915,
                    0x1eba19d33bf05a6d,
                ])),
                Felt::new(BigInteger256([
                    0x8c8f256e0f60fdf2,
                    0xae3e410f173db61a,
                    0xcba94648995c4854,
                    0x0381687619388fcb,
                ])),
                Felt::new(BigInteger256([
                    0xd44ad5dc0081cd26,
                    0x4b785ae4dc2e1b17,
                    0x7d7971e83b8bc60f,
                    0x261c66c96714f78f,
                ])),
                Felt::new(BigInteger256([
                    0xb2b2a81b38049c07,
                    0xb3d7f83247e234e1,
                    0x2c74ca97ba317ff2,
                    0x2a747173e2a843a7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xda54242dd4bea10d,
                    0xb53b3b8affd7985d,
                    0x162b8481437e895b,
                    0x20428de050cf998b,
                ])),
                Felt::new(BigInteger256([
                    0x97f62c7899c5a172,
                    0x1b107c607477534e,
                    0xa7920f18f5d77714,
                    0x16d2f62281fbdde7,
                ])),
                Felt::new(BigInteger256([
                    0x3d40cef4d6a8925b,
                    0x7f4f4f968828a6f8,
                    0x4765df9b2888cf5b,
                    0x0155736402f5b325,
                ])),
                Felt::new(BigInteger256([
                    0x6785a3cb3917f312,
                    0xc15e4bff1263de33,
                    0xf7b90b1f9f4f57c1,
                    0x31ec3ee3e22fa9f9,
                ])),
                Felt::new(BigInteger256([
                    0x0983dfc198618a38,
                    0xccd7be8498520e20,
                    0xda47835708ff1f29,
                    0x1fd81f143389153f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3df97f1c9dbe77ba,
                    0x4f37340fd005ce72,
                    0x6a3e2f9677e673c1,
                    0x20f38234912b37e3,
                ])),
                Felt::new(BigInteger256([
                    0x8d764b06e1c8887a,
                    0x4e4666db4d32bdd8,
                    0x8242d28053b3317e,
                    0x201955e0d0e64817,
                ])),
                Felt::new(BigInteger256([
                    0x23daa01fb37ac590,
                    0x14719995d56abdc4,
                    0x53b12a2884f7292d,
                    0x2b35da799b9bac68,
                ])),
                Felt::new(BigInteger256([
                    0xfab50a869299b179,
                    0x02f8d2cbbc3b8214,
                    0x6c102856bb031b53,
                    0x1e5519e507e072df,
                ])),
                Felt::new(BigInteger256([
                    0x790e086cf428430c,
                    0xa4addb738176ba44,
                    0x1c2cec41d236fbb4,
                    0x1fe3de9c72f36be1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2993d39836fd6d5c,
                    0x4b5879a7366b4162,
                    0xfc8d5145d43e0aa7,
                    0x18e739261778077d,
                ])),
                Felt::new(BigInteger256([
                    0xf88c0f4332db6b78,
                    0x363af712c2d41ba7,
                    0x2d43c3cba480de7b,
                    0x0af96533c8c67669,
                ])),
                Felt::new(BigInteger256([
                    0x46a9b66fe3b44d79,
                    0xce62bb2a68643d81,
                    0x816f3546c543184b,
                    0x1c9ea590ceb7f7ff,
                ])),
                Felt::new(BigInteger256([
                    0x5d72133f47af9245,
                    0x29a57ecc5648e6fa,
                    0x40888c62f46c95f6,
                    0x3ec2429f8dd1725a,
                ])),
                Felt::new(BigInteger256([
                    0x16471f9e537f5cfe,
                    0x549e802a38e536e5,
                    0xc433058bd52d027d,
                    0x3b81010a76227db3,
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
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcaa571325d1cf9cc,
                    0x518c5a34b5486ec0,
                    0x319a64d8a05d8d7f,
                    0x0ca51914f4716507,
                ])),
                Felt::new(BigInteger256([
                    0xcd4ca1bf7af48133,
                    0x41ee2f13c6e1646f,
                    0x6efb6a504fc435ba,
                    0x2879deff49c0b023,
                ])),
                Felt::new(BigInteger256([
                    0x68530475614cb90b,
                    0x2f787c2506984b5c,
                    0x9f5f93439ada6436,
                    0x32d28250fb038022,
                ])),
                Felt::new(BigInteger256([
                    0x4acf135fd5115a0b,
                    0x515799d0e3b637c0,
                    0x5b1f9f7aa03a8dab,
                    0x267fa3f344dfff5b,
                ])),
                Felt::new(BigInteger256([
                    0x15d310da592dc234,
                    0x7181b88f2ca3dc1d,
                    0x9ccfdcaaa50aa325,
                    0x18332521f2ac9744,
                ])),
                Felt::new(BigInteger256([
                    0xb6d8a4a3beae6afd,
                    0x6625b0c3357f5d6b,
                    0x2b63a948188082f1,
                    0x027ce018087cdb4d,
                ])),
                Felt::new(BigInteger256([
                    0x5ed3dc6faffb41e5,
                    0x61440050199ee9fd,
                    0x17f6c3e471d4a1be,
                    0x1883e9de63167f6b,
                ])),
                Felt::new(BigInteger256([
                    0xd9ba3ae774aae956,
                    0xb02e732fde3f94cf,
                    0x8042dbc2f8145712,
                    0x04874435fcbf77fa,
                ])),
                Felt::new(BigInteger256([
                    0x62784d545fb993f2,
                    0x92f1ecf5ece7816d,
                    0xa0ebb72f00627b3e,
                    0x25d01ef31bbc85ca,
                ])),
                Felt::new(BigInteger256([
                    0x2d93c67265f7fa0b,
                    0x196fda1b280e694b,
                    0xd236f089720cd6ab,
                    0x3bdd89ca88b3943a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf03ac50009fa0be9,
                    0xf50c804d48c90c61,
                    0x77b7532f9c094d89,
                    0x31c4801115396a7c,
                ])),
                Felt::new(BigInteger256([
                    0xf808e63573b5fd3f,
                    0x7bb77801b7f3bdac,
                    0x06f66dab486662ed,
                    0x2db1f7e9360b6f5a,
                ])),
                Felt::new(BigInteger256([
                    0x038bb435e429c5c8,
                    0x62929fec3ea7bb3f,
                    0x2542dd040d65aaa4,
                    0x2dafe3b5e674822c,
                ])),
                Felt::new(BigInteger256([
                    0x21274ab3bb60dfc3,
                    0x53b726bc3f786fbb,
                    0x7d457d673a57b92d,
                    0x0adefcb801906cd2,
                ])),
                Felt::new(BigInteger256([
                    0x4539e6b693987169,
                    0xcbb6b1cd891b5908,
                    0x125b7974baf43700,
                    0x3b8fb6cfaf311b39,
                ])),
                Felt::new(BigInteger256([
                    0xcf9260c2f187fe14,
                    0x648ce409b1fbbb74,
                    0x1426b1017cd05d09,
                    0x1d2971d630ca199f,
                ])),
                Felt::new(BigInteger256([
                    0xfc5d9d0d59c13e53,
                    0x63ebadc33502af86,
                    0xd2187a7016d38952,
                    0x3d09df47c2680059,
                ])),
                Felt::new(BigInteger256([
                    0xf466e7d468656cfe,
                    0xd2ec75ac06335367,
                    0xd20042175e282b94,
                    0x278a47d6e86c2d3c,
                ])),
                Felt::new(BigInteger256([
                    0xd5df75660fb3726d,
                    0xee9adde3044a375e,
                    0x76275b98b53311a6,
                    0x08ebb0c6e790dcda,
                ])),
                Felt::new(BigInteger256([
                    0x32c717bbd8f7dfc8,
                    0x2a9032db296826b8,
                    0xefcf90907f3412b7,
                    0x0e3811e1da30a136,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6d551f4b24dd43c0,
                    0x5330fa9c1ec32ddd,
                    0x66ef8f72c11033f6,
                    0x2739347234f776f2,
                ])),
                Felt::new(BigInteger256([
                    0x2dd01b9422b6448a,
                    0xe75a61ba99628f7a,
                    0x8dd4304b8198895d,
                    0x2f5c6ac069002f24,
                ])),
                Felt::new(BigInteger256([
                    0x1f365f8980a770f3,
                    0x1b5fd184da40fa1f,
                    0x074c12cbc630a307,
                    0x1c55a4752510335b,
                ])),
                Felt::new(BigInteger256([
                    0xb636e11e0864a9cb,
                    0x4cc57385ee026845,
                    0xd9549fb934811dab,
                    0x3441b98824a9daf1,
                ])),
                Felt::new(BigInteger256([
                    0x7dcd69f849d4e6dd,
                    0xf54cee5328bd88da,
                    0x1f110da88936594a,
                    0x1b6a6c209b3bba92,
                ])),
                Felt::new(BigInteger256([
                    0x50a40c6d385b48e9,
                    0xe08e956c72bacc48,
                    0x263c89ceff9d6221,
                    0x2b8740f6ba1e9ef5,
                ])),
                Felt::new(BigInteger256([
                    0xf171996e9c8e7684,
                    0x6c3154cf576d54e7,
                    0x62eaab7bfc427bed,
                    0x3727e478c924e594,
                ])),
                Felt::new(BigInteger256([
                    0xd810f1fe6800576d,
                    0xa4f8a27ce74d5335,
                    0xc8013923b554026f,
                    0x133ae6e9482a0574,
                ])),
                Felt::new(BigInteger256([
                    0x2cd25b707a0564df,
                    0x54f5ade1d5cda1e2,
                    0x0f3c92dbe9fb14d4,
                    0x34eca2aa27b0acee,
                ])),
                Felt::new(BigInteger256([
                    0x0f689c13eb7a4a34,
                    0x24f32386dcae4a4c,
                    0xe61763ce98cae33c,
                    0x3cb1a082ce8d4aa5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x92057fc1eff24a8b,
                    0xad7a698d342b17f4,
                    0xf2c36e488b8df40c,
                    0x1fb8ce625deaa03b,
                ])),
                Felt::new(BigInteger256([
                    0xfe1c25a4792cf503,
                    0xf8bcc7699aaba7d6,
                    0x43f5f5973cb3d877,
                    0x333cba3e49b32117,
                ])),
                Felt::new(BigInteger256([
                    0xf463e1bc6534acdf,
                    0xd3a24b450755a084,
                    0x3e26304eb93f1e4a,
                    0x38bcca658bd03223,
                ])),
                Felt::new(BigInteger256([
                    0xd7a42689dd18116c,
                    0x01fa6e17053e86cc,
                    0xbdbd3ad70a4fa279,
                    0x2f223728925a7e9f,
                ])),
                Felt::new(BigInteger256([
                    0xdc14523f585c189b,
                    0x946491b9e3506cb0,
                    0xc69887617201c957,
                    0x0f20063e74b32e3b,
                ])),
                Felt::new(BigInteger256([
                    0xf2d9787bb0bdc5aa,
                    0xacd529a028fae906,
                    0xcc3dc0d947529377,
                    0x1f44b8c38b6340d0,
                ])),
                Felt::new(BigInteger256([
                    0x7db52fefdb7aade3,
                    0xaa69f3741521fb9d,
                    0x3be89893fded0820,
                    0x305b204222e470be,
                ])),
                Felt::new(BigInteger256([
                    0x80a5ceac42d42976,
                    0x1fed002002a710db,
                    0xf9e1ca84fce5be67,
                    0x28422ce0a3709e55,
                ])),
                Felt::new(BigInteger256([
                    0x632356f3dbe40c22,
                    0x2195c9bb6258f495,
                    0x798999efba469ab3,
                    0x3a120a3c9d644212,
                ])),
                Felt::new(BigInteger256([
                    0x1e239a5c020f0e93,
                    0x65b9992cd3697eea,
                    0x6db76d7f681e8145,
                    0x08cfe47696fbf76e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf2b4388151354668,
                    0x3e1d03cae3129f61,
                    0x08e22622e9d9419e,
                    0x155ee224b5bed572,
                ])),
                Felt::new(BigInteger256([
                    0x6d7c8154eab8d418,
                    0x16b596a7bcf78da3,
                    0xabf3838b8e086fdf,
                    0x1222c4d0468ba099,
                ])),
                Felt::new(BigInteger256([
                    0x8d372ecabd2b90f6,
                    0x9ad2d0c6e234af22,
                    0x7d58e0e16f6237bd,
                    0x1b0d68b09f514435,
                ])),
                Felt::new(BigInteger256([
                    0x95f6e88aa63d0f08,
                    0x0e95e47b7bcc9e18,
                    0xfe286892218aecad,
                    0x24db808d09066d7d,
                ])),
                Felt::new(BigInteger256([
                    0x35f7638ef6b48a08,
                    0xd5df3fb5dbcc36f2,
                    0x019c8e08d5085063,
                    0x2ab28a8eb0a50329,
                ])),
                Felt::new(BigInteger256([
                    0x606685088f93255d,
                    0x8fbf4e9859c36c0e,
                    0x67cec2313bbc6ade,
                    0x00fb3c2cf483da5a,
                ])),
                Felt::new(BigInteger256([
                    0x9a052f39e4f2bd29,
                    0xc76b1eec478fa546,
                    0x35ad4b03981a3738,
                    0x385af84835bb075b,
                ])),
                Felt::new(BigInteger256([
                    0xf898c3459a61c08c,
                    0xf3ee2cec44a5da9c,
                    0x529c570419ff1b6a,
                    0x0b6415ee2dc2c85e,
                ])),
                Felt::new(BigInteger256([
                    0x85f6b402a9cce8a5,
                    0x5d880145993d32c5,
                    0xc73be4a956899bb1,
                    0x08b53f081d236dce,
                ])),
                Felt::new(BigInteger256([
                    0x1c08a0f0b4d2d9a5,
                    0x67fc331aa201cb18,
                    0xa459bee0648a5769,
                    0x20773e2f5c25704b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xda65c19213914fac,
                    0x2fe98e5931a3e5cb,
                    0x0fb0b3a1b30386d7,
                    0x1788c31d0468d04c,
                ])),
                Felt::new(BigInteger256([
                    0x0995e90ba7f29bb8,
                    0x7739678767c21e15,
                    0x9722dacecf1b03e0,
                    0x303c40eccfe06635,
                ])),
                Felt::new(BigInteger256([
                    0x0bb95c802d6252c0,
                    0xd5151a9e92e445bb,
                    0x6e35d815b63cfe3b,
                    0x1109941124ef5c88,
                ])),
                Felt::new(BigInteger256([
                    0xb0621eb3baef29f3,
                    0xa0f1ba83f33a39cf,
                    0x0fbc971bf8c9562e,
                    0x2d9a5f09b0806a88,
                ])),
                Felt::new(BigInteger256([
                    0xdb177cb3f1126722,
                    0x30bc6ce9d1c26102,
                    0x932619f6c590b30f,
                    0x161ad0c50f8298d0,
                ])),
                Felt::new(BigInteger256([
                    0xa9323c8e4e1d6894,
                    0x5477a5f2f7e3ad3f,
                    0x1b83234a673fd1c0,
                    0x17dd51a40a0acb31,
                ])),
                Felt::new(BigInteger256([
                    0x864fb5e862ca5e6a,
                    0x043754d7078c237c,
                    0xaff96b6850711dbd,
                    0x02a10a020a395a2d,
                ])),
                Felt::new(BigInteger256([
                    0x5a5e1d3b8eb35cb0,
                    0x9b64a6a2845d9f99,
                    0xa196f489c8766356,
                    0x13dac10ffda43d99,
                ])),
                Felt::new(BigInteger256([
                    0x8858735cf8d2a62a,
                    0xa6644aed15dea63e,
                    0x733c28f46f626022,
                    0x282ceb056727458e,
                ])),
                Felt::new(BigInteger256([
                    0xac302d033a19e52a,
                    0xd44260dd30065d46,
                    0x042f9f87ed7c3b08,
                    0x1a89f570fde95704,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x314b546b42451d3d,
                0x40e4797c7a3e138d,
                0x22a8ea9fea2e3a43,
                0x1a9872f19842faa8,
            ]))],
            [Felt::new(BigInteger256([
                0xf9da7f2a9d5935be,
                0x1bb720cc2f8ce055,
                0x9fcba9ce6b7707cb,
                0x20a2bc3acf589712,
            ]))],
            [Felt::new(BigInteger256([
                0xa37d12d37b86ce90,
                0x6fa38cdd56a3ee01,
                0x5e8beba23d4d4228,
                0x1220347852758ed8,
            ]))],
            [Felt::new(BigInteger256([
                0xaacba291b1b67121,
                0x4d9a1d80d08356e8,
                0x51f5d5de9bc8e579,
                0x218402b2def65180,
            ]))],
            [Felt::new(BigInteger256([
                0x62b1a5a5724bf004,
                0xa4ba203f1240a34d,
                0xb16670cfb1840a5c,
                0x127ee2b8f65c8c62,
            ]))],
            [Felt::new(BigInteger256([
                0x369c9478aeb44320,
                0xd36a833009129782,
                0xcfd0665fc2286a2b,
                0x2987d626e205523e,
            ]))],
            [Felt::new(BigInteger256([
                0x58e36713a365397b,
                0x608f95d98a7e310f,
                0x32cc7d75cadc67f8,
                0x03b43fffa2bb0e57,
            ]))],
            [Felt::new(BigInteger256([
                0x58adb8cdf549d7e6,
                0xda5eea3972da019d,
                0x563b22174200ae4d,
                0x0f57b450f603f488,
            ]))],
            [Felt::new(BigInteger256([
                0xe7ad6d8fbe53338c,
                0x7f1d5343c0e60356,
                0xdc08beeb1be91c6e,
                0x3ab66d934d42bca2,
            ]))],
            [Felt::new(BigInteger256([
                0xd1d384a536258111,
                0x0a5d75906e533711,
                0x4acc11b0aeee494b,
                0x1485b8d2567d5c50,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 10));
        }
    }
}
