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
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 48];
        for chunk in bytes.chunks(47) {
            if num_hashed + i < num_elements - 1 {
                buf[..47].copy_from_slice(chunk);
            } else {
                // If we are dealing with the last chunk, it may be smaller than 47 bytes long, so
                // we need to handle it slightly differently. We also append a byte set to 1 to the
                // end of the string if needed. This pads the string in such a way that adding
                // trailing zeros results in a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 48];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 47 {
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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger384;
    use super::*;
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger384([
                0xf0c9cdc9d33695ea,
                0x4c1490327c945bad,
                0x13c47477b50839cd,
                0x558a9f1feeaddbe7,
                0xe4f47de73d45b85a,
                0x0c2df3f8f7ec6050,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x9ca97ee0afe53569,
                    0xd23b735512be9695,
                    0x4efe358105478ddf,
                    0x92c4c52e2971119e,
                    0xc6a17bb015075d32,
                    0x0973d869153a38f6,
                ])),
                Felt::new(BigInteger384([
                    0x7b024e323bb292d9,
                    0x5e42e7b7815e37b1,
                    0x7dac026174fad54b,
                    0x1d211a26a4163e6b,
                    0x4ee6c4c59e9a82a1,
                    0x139c1f118fb7f0ca,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x925a7fe76ad393fd,
                    0xf5f4df6bac336659,
                    0x4ec73bc85fae4bca,
                    0xf64835493af9fff2,
                    0xb597ef43449bfb94,
                    0x17353552a144b9f8,
                ])),
                Felt::new(BigInteger384([
                    0x1a9f6236a6e870ed,
                    0x7011b6f87581443a,
                    0x66fd4792ba37c985,
                    0x2a0905603702a1df,
                    0xacbbb63415439094,
                    0x144cab552bcfe923,
                ])),
                Felt::new(BigInteger384([
                    0xc77836c4a4dd6b81,
                    0x36f62aec294a79e7,
                    0xceab4a37b1daede0,
                    0xcb46c7ed5efdabd6,
                    0x8f7e58bd107dcc25,
                    0x036efc00d649b50b,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x1f6035a486259e92,
                    0xbbacbc4c59d02a61,
                    0xb5b94158a5c341df,
                    0x65861e38c153d901,
                    0x9f9eaad092891d9d,
                    0x0a4f036f0d83fbe6,
                ])),
                Felt::new(BigInteger384([
                    0x349d19b09ae79c99,
                    0xddf002e33241da4a,
                    0x0424b97c85222b7a,
                    0x0a53f7b68f31730e,
                    0x6fc915016afa08b9,
                    0x05b0927915a7444b,
                ])),
                Felt::new(BigInteger384([
                    0x6a3475097ef1e478,
                    0x6a61d53240437dd4,
                    0x9d8f70933459bd9e,
                    0x3e97ae3a17b94f53,
                    0x80a277572fb7d4da,
                    0x10a2a5461247455c,
                ])),
                Felt::new(BigInteger384([
                    0xa0897104ff7b86df,
                    0x9ee6ff0c7a13b4cf,
                    0x33eed195734a51b8,
                    0x813663445b0d071f,
                    0x1d8fb6e9a7b0e774,
                    0x0ef1afb954c5e24c,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x3aa2af110779af10,
                    0xfd90c9f36145d37b,
                    0xdc7058d5cdc05e0a,
                    0xc695711595110c82,
                    0x9bb98537f6d66338,
                    0x0fbe6c1295043db4,
                ])),
                Felt::new(BigInteger384([
                    0xb96ccea7ae3c7a94,
                    0x312527e176bf784e,
                    0x25f694ce7bf05ce4,
                    0x8a12a67f4bf802f4,
                    0xa57e51504a5a7916,
                    0x0302f67762986111,
                ])),
                Felt::new(BigInteger384([
                    0x0585ac764fc15527,
                    0x2ca521cc8af5a309,
                    0xdbcd9d13c62ba567,
                    0xb640e553d9f542ac,
                    0xce02ad1aa2b73dc0,
                    0x020b3daa241cf2a2,
                ])),
                Felt::new(BigInteger384([
                    0x0ca29ebc267d3606,
                    0xb69f82febb6d3186,
                    0xabb69a043711b17c,
                    0xdc12e1d07493983c,
                    0xdc712323c161d761,
                    0x04e602a889b6b3fa,
                ])),
                Felt::new(BigInteger384([
                    0x4f97f9c808763a9c,
                    0x8a7262c38c3839d8,
                    0x343f14d6a1c7e5bc,
                    0x9ef2c3687d39c555,
                    0x5f248d86c6732d44,
                    0x0762f31b1c16a313,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x603667b601776434,
                    0xc02f2a2ac134c05b,
                    0x0645e9e626236a08,
                    0xe974551d4afcf41c,
                    0x640c3066b2e96d24,
                    0x14430b5ba48a646b,
                ])),
                Felt::new(BigInteger384([
                    0x4335342e9414f8b2,
                    0x27796d6789183dd9,
                    0xa20a68e67005a635,
                    0x3d8de17c286d1e60,
                    0xa9b393495332fed8,
                    0x110ca191d40942c6,
                ])),
                Felt::new(BigInteger384([
                    0xadfc98d382080ba7,
                    0x6b463eaf77db087a,
                    0x046ef38a1addd68e,
                    0x494e704126ebfeff,
                    0x661c0613b6be3756,
                    0x0a54e643486dfe27,
                ])),
                Felt::new(BigInteger384([
                    0xed0fc9dcb0b27e6a,
                    0xfa15b20171954d26,
                    0x699e14698ff2f184,
                    0x062cd4ecc7c66b98,
                    0x3acee364082210f9,
                    0x1848c603506e50e6,
                ])),
                Felt::new(BigInteger384([
                    0xc96f736b2e53c132,
                    0x033dae38e8f32bfd,
                    0xdec45c41abe96855,
                    0xe0f70c85fbbd6a21,
                    0x3a2b8a595e503adb,
                    0x18d13c51bb11ce93,
                ])),
                Felt::new(BigInteger384([
                    0x0be1b9d04f2e429d,
                    0xf3bc551ad3d3db69,
                    0x6d3ba49526993ae9,
                    0xc9df7ddcd79d2065,
                    0xc07981447b30a614,
                    0x05e0a670543506b3,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xa61a1febb7f0e442,
                0x2c474c413f8f7bf6,
                0xf55e850ab74d59bc,
                0xd78cad5449cb117f,
                0x911083fd6d02c575,
                0x1583bc8ef128d2b0,
            ]))],
            [Felt::new(BigInteger384([
                0x0800b42757eac23e,
                0x533a0ccdc1455da0,
                0xff91db07769d03ab,
                0x079e2492453df45e,
                0x66981221028f79e6,
                0x039fa5e4d20c2be7,
            ]))],
            [Felt::new(BigInteger384([
                0x064f34238de07462,
                0xe6ac094756b35ac8,
                0x89b373c1f266b877,
                0xd6f240f1a98dd491,
                0xbb7cf350674adbb0,
                0x09cedf304c7773d4,
            ]))],
            [Felt::new(BigInteger384([
                0x09d93c25d3bd597e,
                0x4158151a5f872993,
                0x2111f3dd78eab4ec,
                0xdfb3cae4cb77734c,
                0x71a1d44e4c62d9ef,
                0x0faac91e454315b0,
            ]))],
            [Felt::new(BigInteger384([
                0xde415d1c80142488,
                0xb42d571c6d182ba0,
                0x11c78533b718d84b,
                0xb009faaf63c8ca89,
                0x102fedbcd3617f8f,
                0x18948a067b856e38,
            ]))],
            [Felt::new(BigInteger384([
                0x7f39a5acb275a13e,
                0xcccf3918c94168ac,
                0x206ba5fc30a8cb3f,
                0xc1b0ef09c4460a29,
                0xb861c2fac190b8be,
                0x0d108a1389537576,
            ]))],
            [Felt::new(BigInteger384([
                0x7d90124b0d5905bd,
                0x172677f6e039faf2,
                0x8cd20f299501a42a,
                0xe24c2ce9d6e920ff,
                0xcd6b1c70cbf31626,
                0x0044bf7f70cf0ffb,
            ]))],
            [Felt::new(BigInteger384([
                0x2caaa8a41a2588ab,
                0xaf5cba83d450ece6,
                0xe9015ac94df2807a,
                0x0fa269d92119b477,
                0x2059feb244ce0315,
                0x047b4fceb97e6d3d,
            ]))],
            [Felt::new(BigInteger384([
                0xea4f16e22e71c0b8,
                0xdd58f1b93746f417,
                0x3da0ddf5859a3c5e,
                0x5aeeedc92b853ce3,
                0x63cfb2ee1d552587,
                0x126f1c3127aba92e,
            ]))],
            [Felt::new(BigInteger384([
                0xe5dafa9629526eb1,
                0xe7c6a88536ab5fa4,
                0x6ff0c90dd2ce487d,
                0x3e333fee423dce15,
                0x1a9785f73fea4eba,
                0x117fa2fe6a4ef537,
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
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xa61a1febb7f0e442,
                0x2c474c413f8f7bf6,
                0xf55e850ab74d59bc,
                0xd78cad5449cb117f,
                0x911083fd6d02c575,
                0x1583bc8ef128d2b0,
            ]))],
            [Felt::new(BigInteger384([
                0x0800b42757eac23e,
                0x533a0ccdc1455da0,
                0xff91db07769d03ab,
                0x079e2492453df45e,
                0x66981221028f79e6,
                0x039fa5e4d20c2be7,
            ]))],
            [Felt::new(BigInteger384([
                0x064f34238de07462,
                0xe6ac094756b35ac8,
                0x89b373c1f266b877,
                0xd6f240f1a98dd491,
                0xbb7cf350674adbb0,
                0x09cedf304c7773d4,
            ]))],
            [Felt::new(BigInteger384([
                0x09d93c25d3bd597e,
                0x4158151a5f872993,
                0x2111f3dd78eab4ec,
                0xdfb3cae4cb77734c,
                0x71a1d44e4c62d9ef,
                0x0faac91e454315b0,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 282];
            bytes[0..47].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..47]);
            bytes[47..94].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..47]);
            bytes[94..141].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..47]);
            bytes[141..188].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..47]);
            bytes[188..235].copy_from_slice(&to_bytes!(input[4]).unwrap()[0..47]);
            bytes[235..282].copy_from_slice(&to_bytes!(input[5]).unwrap()[0..47]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0x27c7d5c2ff3da7f3,
                    0x7b3c48c6ea64e29a,
                    0x71fa18787e4cdf81,
                    0x031e1707d3c9f4f9,
                    0x1b487ac5ae68d140,
                    0x11f2ff678799f16e,
                ])),
                Felt::new(BigInteger384([
                    0x56a331ab7a74a2e4,
                    0x6466b7fdfb4bae63,
                    0xea06e4b9aa944804,
                    0xfa9c4b9ec762297f,
                    0x11d52eac50f31c38,
                    0x15110839f23196fd,
                ])),
                Felt::new(BigInteger384([
                    0x453d35fb4dbdd2f1,
                    0x0c565d54940c8f5f,
                    0x5a7d4e8a0082a44c,
                    0xa57bac12fc75799b,
                    0xac6fcb6b2eb491c7,
                    0x18dbdcae99bb0f45,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb22806e36230d42b,
                    0xe1e0dba6b531a537,
                    0x912cfcfeabab5184,
                    0xb4672b01e31103cd,
                    0x0295a9916e73f551,
                    0x08020db701df6400,
                ])),
                Felt::new(BigInteger384([
                    0xa8c8d4deec6b67fa,
                    0xe9982a4489fb2fba,
                    0xdd72560edf6541bd,
                    0xc2b96e384db87061,
                    0xf06b394b219bf1a7,
                    0x15b7ba52cc6c732d,
                ])),
                Felt::new(BigInteger384([
                    0xe2d564959f81a112,
                    0x4f5ac33fe693c620,
                    0xb9d8235cce939700,
                    0xa1fb7e5ed9ad2533,
                    0xd69622a28fc9c892,
                    0x14f5b32e190ff7e3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xf388cef3649aaf8f,
                    0x750a2d1e3553ef61,
                    0xd7635ac0a8de3aaa,
                    0x8a3dcd9fabdbf909,
                    0x7b257e6b4afb190f,
                    0x0235183b9f87d290,
                ])),
                Felt::new(BigInteger384([
                    0xee936ef53304d518,
                    0x48b7afb236bf15cd,
                    0x75932f98738ad60c,
                    0x2f9067db07d20536,
                    0xddb72a6844d4a9fc,
                    0x1338469c12fa68d1,
                ])),
                Felt::new(BigInteger384([
                    0xc6bc3d4509f852af,
                    0x747a2fa22290ee5f,
                    0x32fbf35a80b7fdac,
                    0x450b5924fd4a90e6,
                    0xc4ebb265a0fe91b5,
                    0x102508c47f4f7d6d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x141bff94bc23e552,
                    0x75d99218cfc47bea,
                    0x13ee45bf39a110a8,
                    0x8695b8ced7abb7ed,
                    0x065e3599508c6160,
                    0x14a30b446da3dc99,
                ])),
                Felt::new(BigInteger384([
                    0xd028a4499ba61756,
                    0x6488bcf64d42886e,
                    0x8819a37f208ce758,
                    0xd2c5be672ab6fd56,
                    0xb9af7413555220b8,
                    0x15b03ea354fee02d,
                ])),
                Felt::new(BigInteger384([
                    0xd8a89ec06531ec48,
                    0x6ee3bc1e25f45b20,
                    0xb667692919aedb1a,
                    0xe907acd5ff5e3168,
                    0x48585da3b362a62b,
                    0x114c1a5ef508a307,
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
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x4faa3d69c770c872,
                0xaea15e1c1715205d,
                0xe81ca67a3c01df89,
                0xda4777afb0977295,
                0x43562570a7792591,
                0x0bddc07ba086ca7c,
            ]))],
            [Felt::new(BigInteger384([
                0x83c74057ee1e328c,
                0xfc27c92c746c9b13,
                0xc146a3c962f3341e,
                0xb4a4cc1416f186a3,
                0x7e7b5dc8dc8e02b4,
                0x18ae694daddbe877,
            ]))],
            [Felt::new(BigInteger384([
                0xeed97b2da1982cab,
                0x13900c73dd4ff38f,
                0x18c1ab12a670183f,
                0x9a62431abd737c67,
                0xd2acb382ed82a7e9,
                0x0b9155b1f851d235,
            ]))],
            [Felt::new(BigInteger384([
                0x48ef429ebcfc939a,
                0x0bee0b2fe0535f7a,
                0x840dad25867ae6d3,
                0x79748d021ab6c12d,
                0x722eb7e3d2a9ce96,
                0x079d407244ab9299,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
