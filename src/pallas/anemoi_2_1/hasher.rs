//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiPallas_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiPallas_2_1 {
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
            AnemoiPallas_2_1::permutation(&mut state);
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
            AnemoiPallas_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiPallas_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiPallas_2_1::permutation(&mut state);

        vec![state[0] + state[1] + elems[0] + elems[1]]
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
                0xc22d231d12669585,
                0x1b422542b79ed70e,
                0x9eb262313ed2f462,
                0x255cbea2fadde4a3,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xf0298f68fd96cd2d,
                    0xf2b693c0cea714d5,
                    0x7443bee7bf61818b,
                    0x2a2f0d3da7f0b761,
                ])),
                Felt::new(BigInteger256([
                    0x3e586261b3b07905,
                    0xaf5ce99b31528f98,
                    0x422715f921258246,
                    0x0df141e930e05d66,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x74a0c93b6f38aa6a,
                    0x77a6d78089adedaf,
                    0xc2708b4db44fd771,
                    0x1548ccb78ae06176,
                ])),
                Felt::new(BigInteger256([
                    0x46bc283fc1f3d29d,
                    0x15c1ec3cc342db78,
                    0xddb07033ef113f7d,
                    0x20755473d05e9968,
                ])),
                Felt::new(BigInteger256([
                    0x14ccdd6c1fbaf75a,
                    0x5dd7a144cef3c840,
                    0x5e12dd7c7acf27ef,
                    0x18997ab906f7cfea,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x706226ee2891ac21,
                    0x3ddea37e07a94cd7,
                    0x598352775744ac4c,
                    0x07d3ed52f9b2a67b,
                ])),
                Felt::new(BigInteger256([
                    0x34583e5d8743c33a,
                    0xcc1c2d300d3c0236,
                    0xa8abc79a1ea89f4c,
                    0x3aa7cf1ebc05a01a,
                ])),
                Felt::new(BigInteger256([
                    0xbccc6a2899207d2a,
                    0x0006ac4a73658646,
                    0x43128443f4f1a751,
                    0x0ab35a62521aec03,
                ])),
                Felt::new(BigInteger256([
                    0xeb9a538ab0b36190,
                    0x6dc977cb52a161e7,
                    0xe371fab7783f5af6,
                    0x0a9fcad73300e062,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7a68217432636078,
                    0xada1c4628daecf04,
                    0x4929cfc5e16adf53,
                    0x3a18b93cd84b8fde,
                ])),
                Felt::new(BigInteger256([
                    0x5a7a3cd1a2bd2440,
                    0xcd2d000c0902834d,
                    0x4507828a1e5d6343,
                    0x190ede5823c9c320,
                ])),
                Felt::new(BigInteger256([
                    0xe8483f7a0958311d,
                    0xb7d94d4f327188e6,
                    0xc04402cae2b90032,
                    0x2e461ccf251aac64,
                ])),
                Felt::new(BigInteger256([
                    0x3140c03d9c064da3,
                    0xace03454a9fa2051,
                    0x9bfef60c4c243cdd,
                    0x295f69403769dec6,
                ])),
                Felt::new(BigInteger256([
                    0xda21713cd1606575,
                    0xa061e5b4e12c410a,
                    0x7f90aa13cbe5bd1d,
                    0x394ac30450a78c58,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe8e737370c8dabd4,
                    0x05feef0aa98df667,
                    0x262f2f4c8cb9d267,
                    0x35a5a8d99e50207e,
                ])),
                Felt::new(BigInteger256([
                    0xd8bd8bf11e8b09da,
                    0x34569d21f2be928e,
                    0x6ff03a6d8b9486f4,
                    0x3cb2d2679fe4752b,
                ])),
                Felt::new(BigInteger256([
                    0xc87b5702126b7a97,
                    0xb5a76611cb987da5,
                    0x62aeda6269c61fef,
                    0x32f316209779bdf2,
                ])),
                Felt::new(BigInteger256([
                    0xe4c85f2f3533147f,
                    0xbf76f40bab4da53d,
                    0xd7205d75f3753e00,
                    0x033ddd0ac233f415,
                ])),
                Felt::new(BigInteger256([
                    0xa84d9658b704d8c6,
                    0x64498b3b181fc781,
                    0x6c5aa9054615d553,
                    0x3f3fe72ba32d6485,
                ])),
                Felt::new(BigInteger256([
                    0xb903b03f4d28521e,
                    0x108078357c160895,
                    0x3f89f02d6799b7f6,
                    0x056a07ec36290de5,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x431a2ebf944f1030,
                0x067cf48ff16fe29e,
                0x0dbb5ca82a985b05,
                0x08b5b15aab677e7b,
            ]))],
            [Felt::new(BigInteger256([
                0x3eb0d0f4c6db003b,
                0x77895bb0aaef7013,
                0x6e9c52dc116eae17,
                0x2c5f9e40d813c774,
            ]))],
            [Felt::new(BigInteger256([
                0xcebd5d7c4210ae08,
                0x4ce258f3c9601aea,
                0x68fafd9664a051ed,
                0x1d1db809fa5a3cdf,
            ]))],
            [Felt::new(BigInteger256([
                0xae2cd6fafdcc134d,
                0x3939eaafcd1411d5,
                0x225d6d34ddf54097,
                0x05e8b55dd3e44209,
            ]))],
            [Felt::new(BigInteger256([
                0x0e4161bb4be80609,
                0x8ffd7a343c122e65,
                0x2b9552b3264c6597,
                0x3d220fdf7fb292fb,
            ]))],
            [Felt::new(BigInteger256([
                0x002b12bb1dc362c0,
                0xc0d8952951052103,
                0xfb51c1c3adf0678d,
                0x2d19d4cebfbcd120,
            ]))],
            [Felt::new(BigInteger256([
                0x3da77b900d822799,
                0xddce2002280fd2b4,
                0xa60215707500c9d9,
                0x299fc7372085ab88,
            ]))],
            [Felt::new(BigInteger256([
                0x9d278853ef7ba4af,
                0x085573a8b87b2b01,
                0xd06547437b2851af,
                0x3912c406afd3ca26,
            ]))],
            [Felt::new(BigInteger256([
                0x7caa9d3e66431b74,
                0x27b9865f9b28fbf0,
                0xfe2b2f682d9d49df,
                0x078ecf9cda89451b,
            ]))],
            [Felt::new(BigInteger256([
                0x289ebec8f54a0ead,
                0x993d663aa55633b5,
                0x9b7d1322a886a9de,
                0x0abdc12707a4fa69,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiPallas_2_1::hash_field(input).to_elements());
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
                0x431a2ebf944f1030,
                0x067cf48ff16fe29e,
                0x0dbb5ca82a985b05,
                0x08b5b15aab677e7b,
            ]))],
            [Felt::new(BigInteger256([
                0x3eb0d0f4c6db003b,
                0x77895bb0aaef7013,
                0x6e9c52dc116eae17,
                0x2c5f9e40d813c774,
            ]))],
            [Felt::new(BigInteger256([
                0xcebd5d7c4210ae08,
                0x4ce258f3c9601aea,
                0x68fafd9664a051ed,
                0x1d1db809fa5a3cdf,
            ]))],
            [Felt::new(BigInteger256([
                0xae2cd6fafdcc134d,
                0x3939eaafcd1411d5,
                0x225d6d34ddf54097,
                0x05e8b55dd3e44209,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiPallas_2_1::hash(&bytes).to_elements());
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
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xed46edf2868a0ef3,
                0xb31c1796953755d5,
                0x5ab464fefaab04c9,
                0x222b2ae53b0bd7b9,
            ]))],
            [Felt::new(BigInteger256([
                0xaef819247fad840d,
                0xe090a6f00c054882,
                0x2bf26370bffba08f,
                0x316e6f6108567b83,
            ]))],
            [Felt::new(BigInteger256([
                0xb7d7de8c3ad19f78,
                0xd4e2414667d14069,
                0x0afc895a369256a9,
                0x0f576cb3073f2595,
            ]))],
            [Felt::new(BigInteger256([
                0xdd881f35ff70cb59,
                0xfa15105ccbe10a3b,
                0x4bd637fe127d7102,
                0x12ec54a408c0e2c6,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiPallas_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
