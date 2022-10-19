//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, STATE_WIDTH};
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

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 32];
        for (i, chunk) in bytes.chunks(31).enumerate() {
            if i < num_elements - 1 {
                buf[0..31].copy_from_slice(chunk);
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
            state[0] += Felt::read(&buf[..]).unwrap();
            apply_permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            apply_permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // We use internally the Jive compression method, as compressing the digests
        // through the Sponge construction would require two internal permutation calls.
        let result = Self::compress(&Self::Digest::digests_to_elements(digests));
        Self::Digest::new(result.try_into().unwrap())
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
        // This instantiation only supports Jive-2 compression mode.
        assert!(k == 2);

        Self::compress(elems)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;

    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x0e8bdc6064c8cccd,
                0x6752351d0c10a96d,
                0xcb7bc7259352037f,
                0x055cd538e41493df,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x76ee80e2ded8e8e2,
                    0xc68311260a869df0,
                    0x91cd1865ad5b9e21,
                    0x265a69bd86a6d7c3,
                ])),
                Felt::new(BigInteger256([
                    0x00ab4fd1f872b599,
                    0x0b45dc0a5d2d1da1,
                    0x2a78aa91fe92b7b5,
                    0x28e7da3314984da2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf99b9211fdf9c216,
                    0x884d328222e9b9b0,
                    0x780994c3cc0df2b9,
                    0x22c56e379fe39bd5,
                ])),
                Felt::new(BigInteger256([
                    0x8abee4269e512065,
                    0x3820d597e3c2e8da,
                    0xe6490c2c651d63cc,
                    0x00d1eb2c89703760,
                ])),
                Felt::new(BigInteger256([
                    0xc9c43e62266c712b,
                    0x81c5c4f5e5ec4722,
                    0x3609f3ab972d5863,
                    0x064fcca9fc7ff7e0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe5d6add807687838,
                    0xac2e10b64105a353,
                    0x0731519f3d98ef38,
                    0x27259c3fde1b0b97,
                ])),
                Felt::new(BigInteger256([
                    0xf6cb5bd7fc94f650,
                    0x5b85aeac43825173,
                    0x58c37c7dfeb8544f,
                    0x39fe5e94cbcfba78,
                ])),
                Felt::new(BigInteger256([
                    0x2c54a0358f93dad9,
                    0x9db915053f3561c3,
                    0x4186783a1a7373f8,
                    0x2cca3020991db775,
                ])),
                Felt::new(BigInteger256([
                    0xbfedf59462d8649c,
                    0xd177e88351fe22a8,
                    0x7e77ff4cc076a90e,
                    0x1c56c25aafa5ad4d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaa0062b0be28a052,
                    0xe15dcc6760f32016,
                    0xaa810fc1fc016026,
                    0x188150595171be01,
                ])),
                Felt::new(BigInteger256([
                    0x94da2c1545036acb,
                    0x13791ea6e2ae63f6,
                    0xad0175bab1fc7a49,
                    0x1b3a83bca74314e0,
                ])),
                Felt::new(BigInteger256([
                    0x55bc62ced1807b15,
                    0xf40f31d53280185c,
                    0xdfe2c86b23384b1a,
                    0x1c423bd83f504457,
                ])),
                Felt::new(BigInteger256([
                    0xec8083816cc2e2f5,
                    0x4a9aac1065b3a0c2,
                    0x46e55623c966294b,
                    0x261d5e2e4136af77,
                ])),
                Felt::new(BigInteger256([
                    0x3455f585a87f8b47,
                    0x10118ea4d6c74eea,
                    0x34673753e2734982,
                    0x3a8eca4708bef5d6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x375202f91ed53260,
                    0x41ca768731e00557,
                    0xe24944c18a3e4ee8,
                    0x3a75ab8fe1623cd6,
                ])),
                Felt::new(BigInteger256([
                    0xb59b67dda234f057,
                    0xd48d29c3a333e0f7,
                    0xbf233fcb14d45092,
                    0x029b89356c664119,
                ])),
                Felt::new(BigInteger256([
                    0x9db3bb397aa37f54,
                    0x620991e9e68b0da7,
                    0xde401bd731f719f9,
                    0x397d830f2f79b413,
                ])),
                Felt::new(BigInteger256([
                    0x9138281dd56520c3,
                    0xc146c8c594ed9aba,
                    0x5fe124ad1a4d19df,
                    0x089a80db00720454,
                ])),
                Felt::new(BigInteger256([
                    0x5b8669c6eeffdbd4,
                    0xd089df0f6ba3a309,
                    0x41897db680b03d50,
                    0x00353ab112e98cec,
                ])),
                Felt::new(BigInteger256([
                    0x675ffe89ea911729,
                    0x15f5d0c085c0fee6,
                    0x87c883b936e21d0e,
                    0x3bf70c2d3884c442,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x51a17369bbb2bb27,
                0xdb99850aab1227fd,
                0xe4a70ba375235059,
                0x3df72a23d7000352,
            ]))],
            [Felt::new(BigInteger256([
                0xef880bcd36cf6eac,
                0x2330e7e5b145ff12,
                0x6c143e8b74b2b27f,
                0x1710e6b1226869f7,
            ]))],
            [Felt::new(BigInteger256([
                0xa522c557ac7a15d8,
                0xc0c41e10b7ab14a1,
                0xb24bf2ef6316b7ec,
                0x2dd16238edc247e5,
            ]))],
            [Felt::new(BigInteger256([
                0x12f72de93c0d8557,
                0xc51f1613fa86f612,
                0x38f32bcbd8d8f69d,
                0x3113ac8172970b3c,
            ]))],
            [Felt::new(BigInteger256([
                0x0e22e0e746b55f74,
                0x96be9e88164afebe,
                0x43b46bc111fd8bab,
                0x0de228c690015ea1,
            ]))],
            [Felt::new(BigInteger256([
                0x00d7f46935a09923,
                0xbbce6d238458633c,
                0x161b808d089448bf,
                0x338593f2faa6808a,
            ]))],
            [Felt::new(BigInteger256([
                0x8f37beaf89aa7e27,
                0xabba275355ce217c,
                0x1fa4ec4a4064c6c9,
                0x2912a3e213391901,
            ]))],
            [Felt::new(BigInteger256([
                0x8426cee2771cc752,
                0x056ced54080112cc,
                0xac92134720759953,
                0x38649532053264c1,
            ]))],
            [Felt::new(BigInteger256([
                0xa202ed21a6e549b3,
                0x3c70893e0145d942,
                0x42a145c2e5594d42,
                0x1ed46c75c85c387e,
            ]))],
            [Felt::new(BigInteger256([
                0x357f63159b5bc985,
                0x128a74bc32315499,
                0x81f933f097ad96ab,
                0x3f4551c42f52a9fd,
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
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x51a17369bbb2bb27,
                0xdb99850aab1227fd,
                0xe4a70ba375235059,
                0x3df72a23d7000352,
            ]))],
            [Felt::new(BigInteger256([
                0xef880bcd36cf6eac,
                0x2330e7e5b145ff12,
                0x6c143e8b74b2b27f,
                0x1710e6b1226869f7,
            ]))],
            [Felt::new(BigInteger256([
                0xa522c557ac7a15d8,
                0xc0c41e10b7ab14a1,
                0xb24bf2ef6316b7ec,
                0x2dd16238edc247e5,
            ]))],
            [Felt::new(BigInteger256([
                0x12f72de93c0d8557,
                0xc51f1613fa86f612,
                0x38f32bcbd8d8f69d,
                0x3113ac8172970b3c,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x17103b8a5e2a09bb,
                    0xc541f0456b84df1f,
                    0x30a5c3e628c83bbc,
                    0x035b1e2522e8177a,
                ])),
                Felt::new(BigInteger256([
                    0x0703f72e86f3ab43,
                    0x85e5cb01aa1a0078,
                    0x4979ca3746d82802,
                    0x1b1adce1a00d2ed1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x801b3cb60e9d5648,
                    0xb5fe11d28e9125f9,
                    0xa8e8ebdc769a7978,
                    0x10da2c1149eedf71,
                ])),
                Felt::new(BigInteger256([
                    0x4dc0ae839d705edd,
                    0x8eaa96c9128908d0,
                    0xa9c43b8be1d48536,
                    0x227d5cd86cdf6ab0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcfc6d6529f815ee6,
                    0xe1d7d0e02e1efe75,
                    0xd52d056df62060d6,
                    0x3a1946f1c5566716,
                ])),
                Felt::new(BigInteger256([
                    0x6306f8eec3d0d5f4,
                    0x00e8e190220900eb,
                    0x378679015cd3573a,
                    0x20cdb014f15fba7b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xba9b61a8724bee45,
                    0x5caaa5b73cd5bb22,
                    0x12fa704bcd9660d9,
                    0x1661ef49fe58c2eb,
                ])),
                Felt::new(BigInteger256([
                    0x2a390edfa69abb47,
                    0x7d76fc250da75fa3,
                    0x7beedcc0e130d311,
                    0x054f523a911b8e6b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4392259afbe71f58,
                    0x591598d19e0757e6,
                    0x4586a2c0a03ea925,
                    0x167c041c44570e06,
                ])),
                Felt::new(BigInteger256([
                    0x9ae4cde9208c745c,
                    0x5d24bd235bcd6443,
                    0x32b148fce07a386d,
                    0x0a830ddd7dc43596,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdcce194289e2b7c3,
                    0xfcb9c6ee0a47a822,
                    0x8825d49123b19b29,
                    0x2a1fc90e17114f67,
                ])),
                Felt::new(BigInteger256([
                    0x8d08653f8575d75a,
                    0x51490c2e81733ac6,
                    0xb06caff0805367d9,
                    0x02cb8c7f09d36b7d,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xefb6846ff037177c,
                0xba1cbb9306ed082c,
                0x87ea92377ad0d5dc,
                0x0ede33e67b285e6a,
            ]))],
            [Felt::new(BigInteger256([
                0x44fa28e601300c44,
                0x1cf512574a7da183,
                0xddccc5f9a331b74c,
                0x2b91b14d5603d272,
            ]))],
            [Felt::new(BigInteger256([
                0x158a684116751b09,
                0xbf92e84be0fff990,
                0xed4da80389624066,
                0x32a7c3f25b1f57d1,
            ]))],
            [Felt::new(BigInteger256([
                0x3da1050414c88970,
                0xd732e46dd8f50c5c,
                0x0982b127bf77ecd1,
                0x393340439fb5641f,
            ]))],
            [Felt::new(BigInteger256([
                0x6f11f0162e54814d,
                0x42801aa37dd1a8b6,
                0x4b967beff8e7417b,
                0x260618e905584d2a,
            ]))],
            [Felt::new(BigInteger256([
                0x461cd60fd266c4d5,
                0x6ff18b00907327ad,
                0x8940cdb0b78c4561,
                0x2eb5a093705f0909,
            ]))],
            [Felt::new(BigInteger256([
                0x6ebb52fd124e31b4,
                0xba629829748fa9f7,
                0xcce4419a24fce159,
                0x399af56c95056f0b,
            ]))],
            [Felt::new(BigInteger256([
                0x7e027ebe7cff9f70,
                0x981d7f4f048f990d,
                0xf7f7dc79bf7bd63e,
                0x3ab28877165e4dc4,
            ]))],
            [Felt::new(BigInteger256([
                0xc99e582de8cdad7c,
                0xfa65959ec110ade7,
                0xd4e220ad44f0a7be,
                0x39f113bc0a0dc5e3,
            ]))],
            [Felt::new(BigInteger256([
                0xe90149f83df09c5e,
                0xecbedcc5213680b7,
                0x121ae1a5c43c284c,
                0x3d3b4c58cd2ab1f4,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiHash::merge(&[AnemoiDigest::new([input[0]]), AnemoiDigest::new([input[1]])])
                    .to_elements()
            );
        }
    }
}
