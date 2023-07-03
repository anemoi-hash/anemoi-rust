//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, STATE_WIDTH};
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

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 48];
        for (i, chunk) in bytes.chunks(47).enumerate() {
            if i < num_elements - 1 {
                buf[0..47].copy_from_slice(chunk);
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

    use super::super::BigInteger384;
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
            vec![Felt::new(BigInteger384([
                0x15f04f08e89c9bcc,
                0xe02cad85b1219b1b,
                0x9163f2a696ed3957,
                0xcef45007d28aff29,
                0x6fb5147aa162f4c3,
                0x097ed1ba48b79cb3,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x3e2189bebfc6dd6e,
                    0x65667f262e506831,
                    0xc365e5c44a5cde69,
                    0xfcd3b7d20fbfa19e,
                    0x6c53646b586683c2,
                    0x0ca690ac9b049c29,
                ])),
                Felt::new(BigInteger384([
                    0x193ed26205577342,
                    0xf2e6984f35bc01ac,
                    0x9d5e3ba015b59135,
                    0xc76debc27426742c,
                    0x7f30a27ac285803a,
                    0x13a117c240bd36bf,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x05576177e4271f55,
                    0x66b2af3ffcaa62d2,
                    0x72d80f1b8e8bcad3,
                    0x9b8cd412f0781b59,
                    0x9fe64cf460201ea0,
                    0x0b53dec02cbab5ce,
                ])),
                Felt::new(BigInteger384([
                    0xd42465ccd8d1ad5a,
                    0x582887c226dfca88,
                    0xb3642ca404ae8f1c,
                    0x6865ca20f754b3c2,
                    0x32f3b2cd6294b331,
                    0x18e9731871196661,
                ])),
                Felt::new(BigInteger384([
                    0x06c2f0f054941620,
                    0xd12634b153f314a6,
                    0x7a14e30f94a80926,
                    0x5574284a703b4df4,
                    0x95d6809e21a24b54,
                    0x18ca5f0185305548,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x160e70f0da152195,
                    0x37af04ace29ab1a2,
                    0x85cd2d8f9fcf1e1e,
                    0xae9d5378e6f4c09d,
                    0x1a48432f6abea517,
                    0x07725b01ee5f33ca,
                ])),
                Felt::new(BigInteger384([
                    0x12219635124b89a2,
                    0x01f1113c68512315,
                    0xd95cadd9ecf53c68,
                    0x5ef1e6c498161330,
                    0x3af143aae5c4b2d0,
                    0x11b344f5c7a27e7c,
                ])),
                Felt::new(BigInteger384([
                    0x1abc8b647578619c,
                    0x854b7ce946147b74,
                    0x9d9c29b5a347fd22,
                    0x68ba20df1f95b4f8,
                    0x0a8ffb9e64a882e6,
                    0x0424a5b53af27708,
                ])),
                Felt::new(BigInteger384([
                    0x32df725964672423,
                    0x91ae7ababe533ccb,
                    0x67764e32cc3012b9,
                    0x16de3224e10a1446,
                    0x73d9cb94edcf20fd,
                    0x098815501085705d,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x9adca94ae2020d39,
                    0xf04bbec6a34bcbc3,
                    0xec1bec940256067e,
                    0x1ca764a7e25314c6,
                    0xc4ade129e491acbc,
                    0x11ce619c38fd52c9,
                ])),
                Felt::new(BigInteger384([
                    0x61d4ce6c72643800,
                    0x4b17375245b93bf3,
                    0x6c860a27e8f9ae8d,
                    0xb1697baddef4ef6e,
                    0x7ce64116008ec604,
                    0x0820eea7a11f2a76,
                ])),
                Felt::new(BigInteger384([
                    0x6bf7e1bef114c2f2,
                    0x47f6d74b432bb663,
                    0xe3ad667f96e2a0c8,
                    0xcc9132f7ffdda184,
                    0xb9cbd5d4cec531a0,
                    0x01e3db58432f12e5,
                ])),
                Felt::new(BigInteger384([
                    0x3207009a7e772a74,
                    0x937ff8980398fa1e,
                    0xa724c5b48a877274,
                    0xe8e345881c8caa88,
                    0x2a9e17d18c802854,
                    0x14b9a5f7ea7878ed,
                ])),
                Felt::new(BigInteger384([
                    0x70ed3c45bc960f93,
                    0xf444979374fb4b92,
                    0x9266b046df1f6590,
                    0xb6752b70909f5f46,
                    0xdafdea72c1055a11,
                    0x05309f4588480983,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x678f51de3bd57dcc,
                    0x7f473ba6628fd6c2,
                    0xb754af9be093dcda,
                    0xd9a2d21d9865a54e,
                    0xe10bd9a236e0bd90,
                    0x03708204beae4e56,
                ])),
                Felt::new(BigInteger384([
                    0xc685c63f1dd63617,
                    0xb2d4b0c0def417dd,
                    0x02f279919d485f60,
                    0x418d441a0f5159d2,
                    0xe205b9239b2e77c4,
                    0x017da813a794a63e,
                ])),
                Felt::new(BigInteger384([
                    0x9f31a5dc75c3d9b9,
                    0xf605f38b5d7a7f96,
                    0x3aec2b6f1ba2f978,
                    0x6063c1216d4ffba3,
                    0x717fd5afbbe39598,
                    0x0eaeba20d5a54d5b,
                ])),
                Felt::new(BigInteger384([
                    0xbcd0aec29cc1961e,
                    0xb0981aaa9727b701,
                    0x19b75a752a6af0b1,
                    0xd57089b041636004,
                    0x724d5d06f997fc2c,
                    0x19c0356184fcae84,
                ])),
                Felt::new(BigInteger384([
                    0x20b61de4ed60e185,
                    0x9c03b2321015c4b8,
                    0xbaf83dc740b55e84,
                    0x61ffc875266a4710,
                    0xadba51e0ef395c9a,
                    0x0f5ee63919cf00ea,
                ])),
                Felt::new(BigInteger384([
                    0x60624b6d43254cff,
                    0xf03e50e2edf10672,
                    0x709cf072ba112ea9,
                    0xaf0cad714ba967f9,
                    0x4b84e2f849867053,
                    0x12587da8578e245c,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xfd1a73e074e0fc06,
                0xb74fc1f76c908a54,
                0x0de2985c872bee26,
                0x52637494323c398c,
                0xb94b96cbc068a249,
                0x00836ebddd9de996,
            ]))],
            [Felt::new(BigInteger384([
                0x0c70a2aa23ad4b20,
                0x7b595a35c5672b3c,
                0xad2dd4abf68682e8,
                0xc28dd1a4eec32f47,
                0x185454e230f90385,
                0x11929e71fac20d9c,
            ]))],
            [Felt::new(BigInteger384([
                0xe9884f011768bec9,
                0xbae06f8770dc8800,
                0xb645f7a3de9492e7,
                0x12fe10af9e14d84d,
                0xc37a691ed0f7050c,
                0x15823919a5e10319,
            ]))],
            [Felt::new(BigInteger384([
                0x6101b818261bb46c,
                0x972b0a3ad9794d28,
                0xe8dabb328a612774,
                0x0fa3418d0abf19f5,
                0xd5a1dc76baab1d16,
                0x08d115f977a4163b,
            ]))],
            [Felt::new(BigInteger384([
                0x1b1bdcbe9826c7ea,
                0x5d5631b0e17c986c,
                0x61711ea155137bdd,
                0x0a79c698779da25a,
                0xe97ccb33e99f2c6e,
                0x18268f57f6fe4c88,
            ]))],
            [Felt::new(BigInteger384([
                0x66404f53718be739,
                0xbb94f13b90f51735,
                0x817f56bf21bfcb17,
                0x8430a660c73ba52c,
                0x05dfcabab4e26aed,
                0x0de8429fc3486a56,
            ]))],
            [Felt::new(BigInteger384([
                0x1efa0b57ac63f16c,
                0x4a41bad543912ffb,
                0x4ef944c38bba6dae,
                0x8458a77bf2863e37,
                0xb27aae76429d8502,
                0x198a61031631731f,
            ]))],
            [Felt::new(BigInteger384([
                0x8ed9c64a84db4eca,
                0xbbc5608db1441181,
                0xe532e3fb4ce324dc,
                0x2b96efd47a5f7e20,
                0x8ad3077b7c2b430b,
                0x015b66d5261e2820,
            ]))],
            [Felt::new(BigInteger384([
                0xa1f7f5f0c0e5a5f6,
                0xcd472d5a28443075,
                0x3f3d8ff1dda32b25,
                0xdf8d50d7db25a0ff,
                0x70f0e9083f531bad,
                0x027511ff9c66d896,
            ]))],
            [Felt::new(BigInteger384([
                0x94a5a0aa4693e5c1,
                0x68de758e5cec9b5f,
                0x367a5b2ba9bd3548,
                0x30fc70eae3278063,
                0x51bf964a3ada6bdc,
                0x191a516d18acd938,
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
            [Felt::new(BigInteger384([
                0xfd1a73e074e0fc06,
                0xb74fc1f76c908a54,
                0x0de2985c872bee26,
                0x52637494323c398c,
                0xb94b96cbc068a249,
                0x00836ebddd9de996,
            ]))],
            [Felt::new(BigInteger384([
                0x0c70a2aa23ad4b20,
                0x7b595a35c5672b3c,
                0xad2dd4abf68682e8,
                0xc28dd1a4eec32f47,
                0x185454e230f90385,
                0x11929e71fac20d9c,
            ]))],
            [Felt::new(BigInteger384([
                0xe9884f011768bec9,
                0xbae06f8770dc8800,
                0xb645f7a3de9492e7,
                0x12fe10af9e14d84d,
                0xc37a691ed0f7050c,
                0x15823919a5e10319,
            ]))],
            [Felt::new(BigInteger384([
                0x6101b818261bb46c,
                0x972b0a3ad9794d28,
                0xe8dabb328a612774,
                0x0fa3418d0abf19f5,
                0xd5a1dc76baab1d16,
                0x08d115f977a4163b,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 94];
            bytes[0..47].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..47]);
            bytes[47..94].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..47]);

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
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x0f183ee330e80760,
                0x2b3a358c00b8a014,
                0xbe18abc3b76bbb99,
                0x22e3503f4ff44d1f,
                0x7437e59620260622,
                0x04598f027c047176,
            ]))],
            [Felt::new(BigInteger384([
                0x89570178bb493a7c,
                0x1a504f123291d873,
                0x943131497423691f,
                0x8a79b46a68425ac8,
                0x5738821915010480,
                0x0a063fa92a3e7219,
            ]))],
            [Felt::new(BigInteger384([
                0x94a5782689217707,
                0x39323b77b7b8d754,
                0x7212dc2e0a10d87a,
                0x7d61d592e05b9d39,
                0x09d9a261c8a55373,
                0x0c803f97c6fa5538,
            ]))],
            [Felt::new(BigInteger384([
                0xf5549b71920ece07,
                0x752c05c2200f3f92,
                0x866c47f080c25ea4,
                0x4baabab1b3cb3948,
                0x072991b0daf2eb72,
                0x101c9148fdea30fc,
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
