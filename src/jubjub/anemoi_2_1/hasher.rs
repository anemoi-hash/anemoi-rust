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
                0x3dc3160030865828,
                0x9d4c704afa63eb62,
                0xe3ab9369cc7a41f3,
                0x39838be1c0cf295c,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x2f0bc94ead7e5ca7,
                    0x9c286067f9d92b21,
                    0x53d8e4717e52f48e,
                    0x36d3760c9fccdcee,
                ])),
                Felt::new(BigInteger256([
                    0x985565aa98277c71,
                    0x68a4a270c5eb2791,
                    0xb03e9d9d8aec2600,
                    0x490f5569c49b08f7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7b8cd4b59773a04c,
                    0x1739341c6fec5099,
                    0xae69ef87678b8245,
                    0x0c732c2473fdba0f,
                ])),
                Felt::new(BigInteger256([
                    0x9bdb90e9619d3ebb,
                    0x5b241bc4ef36bdad,
                    0xb55be30b6b607ce5,
                    0x4c59aa36083435e2,
                ])),
                Felt::new(BigInteger256([
                    0x43763a3fd1956829,
                    0x93236621fd9266d9,
                    0xee111d5f1b4d9238,
                    0x5bd6a4f5010e5628,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x004fc6a898b69a70,
                    0x394abfc48315581c,
                    0x66d4702b21ce81cc,
                    0x18ef0ef7628caa1a,
                ])),
                Felt::new(BigInteger256([
                    0xf8849d039be6e076,
                    0xb3b0e442d002e2af,
                    0x45154650a638caa4,
                    0x33b8315223a67159,
                ])),
                Felt::new(BigInteger256([
                    0xd930bc26670daaff,
                    0x3170c93795fafa96,
                    0xdb200dfda777dc66,
                    0x38f66e69abb8e749,
                ])),
                Felt::new(BigInteger256([
                    0xcd4440face5ee4a6,
                    0xe229a856fa7c557a,
                    0xb8846bd7d1d583a4,
                    0x1dc22a58a505e6e3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9d2f9ee8fa374ab4,
                    0x79eddaac8658ca23,
                    0x35514a1e68f2bd88,
                    0x0f18b49ad72459f6,
                ])),
                Felt::new(BigInteger256([
                    0xb7bbb533a9474926,
                    0x545b8a9ce28ad0a3,
                    0x4233053b066a8a40,
                    0x5b2c42952e53c588,
                ])),
                Felt::new(BigInteger256([
                    0xae7406ac1fd0f9ae,
                    0x56e1eddb48fefaff,
                    0xb5d5c6c08ad51866,
                    0x6904b378e10aa361,
                ])),
                Felt::new(BigInteger256([
                    0x867a4dbb081e01e3,
                    0xc1623a9e28c421bd,
                    0x00ca4d72b1453862,
                    0x3a91bd189e86d7bc,
                ])),
                Felt::new(BigInteger256([
                    0x4861d7b74300bf6a,
                    0x1a606af17e807afa,
                    0xaa9790b4d3961af0,
                    0x13e38e9b8606cd1c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfa435eca22287a65,
                    0xd63cf88dff6995df,
                    0x93b839438d721b7e,
                    0x406c16d0cfc4aaae,
                ])),
                Felt::new(BigInteger256([
                    0x0a0970defdbbc7a4,
                    0x3d1a2c5f2a492be8,
                    0x709e63cfbe520cc5,
                    0x2f30022011b507a2,
                ])),
                Felt::new(BigInteger256([
                    0xdd3916b2241ca32f,
                    0x2d3704aee1e75d19,
                    0xee48a1ec8cf6bc7e,
                    0x18e71e9e6295be8c,
                ])),
                Felt::new(BigInteger256([
                    0x43103fdd429a1033,
                    0x5b411ae67e39b7ce,
                    0xe428fcca04cfb8ba,
                    0x49af956995be5aa6,
                ])),
                Felt::new(BigInteger256([
                    0x414dddeb2902c2d5,
                    0x9a7ed1587f4db0d9,
                    0xb2817a9fa017b029,
                    0x10d7da6ad19be926,
                ])),
                Felt::new(BigInteger256([
                    0x42da37ede37a37e2,
                    0x72bb2b3fa97febc0,
                    0xa48b6f82b08c1229,
                    0x60d7b34a21cfff63,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x29410f24c26b9ef9,
                0x120c694b4d7aecf3,
                0xb3fc27dd40a5bd15,
                0x165e43d931ea76b1,
            ]))],
            [Felt::new(BigInteger256([
                0xff1cbca354e4622a,
                0xdc9adc9b1d41beac,
                0xb69f767be14a9efc,
                0x2629fedc80826648,
            ]))],
            [Felt::new(BigInteger256([
                0x302d031ce7914abd,
                0x4201d3ec01e1a05f,
                0x4052f31ec35681d4,
                0x16979afc92b601a3,
            ]))],
            [Felt::new(BigInteger256([
                0xb89d1fa84ff1465b,
                0x150d30f4394586de,
                0xc762986d5d8cdc7e,
                0x3aa2efddc2d15109,
            ]))],
            [Felt::new(BigInteger256([
                0x72d4f4af1b3d08f7,
                0x45361a1128b2d5f6,
                0xf3c97a95830bdc69,
                0x73533b1ab048460d,
            ]))],
            [Felt::new(BigInteger256([
                0x758b5eaced3f33bc,
                0xaaccd6a597152777,
                0x0b9bc906fe1aedc6,
                0x256234f348a08fd0,
            ]))],
            [Felt::new(BigInteger256([
                0x6542377dc45e1efa,
                0x1a08206fff0568c2,
                0x3e4b3574e7f08515,
                0x720e87e40e9aa323,
            ]))],
            [Felt::new(BigInteger256([
                0x6975cc85047d5272,
                0x0cc239113293c39e,
                0xb85eb7f75cf76e57,
                0x6cc31c6a62363c50,
            ]))],
            [Felt::new(BigInteger256([
                0xe9e8fd75547d77c3,
                0x3367186bc817f522,
                0xcd8b7e2a679abeaf,
                0x43e30fb1e2005ee3,
            ]))],
            [Felt::new(BigInteger256([
                0x0a2f8e5041650107,
                0x6348e6c39586ed8b,
                0x7a11fb1b5d1e613a,
                0x107458faad7a5686,
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
                0x29410f24c26b9ef9,
                0x120c694b4d7aecf3,
                0xb3fc27dd40a5bd15,
                0x165e43d931ea76b1,
            ]))],
            [Felt::new(BigInteger256([
                0xff1cbca354e4622a,
                0xdc9adc9b1d41beac,
                0xb69f767be14a9efc,
                0x2629fedc80826648,
            ]))],
            [Felt::new(BigInteger256([
                0x302d031ce7914abd,
                0x4201d3ec01e1a05f,
                0x4052f31ec35681d4,
                0x16979afc92b601a3,
            ]))],
            [Felt::new(BigInteger256([
                0xb89d1fa84ff1465b,
                0x150d30f4394586de,
                0xc762986d5d8cdc7e,
                0x3aa2efddc2d15109,
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
                    0x9047f1a62fd94536,
                    0x7473b84d2de3efae,
                    0x8d573c1a0796da07,
                    0x027602e98017d4d5,
                ])),
                Felt::new(BigInteger256([
                    0x1e2b134b02b15eef,
                    0xa5c403035911269a,
                    0x9bc4b101fe12aef3,
                    0x272bbb068d1037a6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3682fed662997141,
                    0xa9a7bbafb8dc474e,
                    0x391f3137a22244b9,
                    0x420a47f4cc8b3104,
                ])),
                Felt::new(BigInteger256([
                    0x12c00ac817a62c91,
                    0x2d7e868be38e69c0,
                    0xed98b11e65cc95a5,
                    0x5d70b58e9f937a6f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdbaeaad5959f8de9,
                    0x614e14ad883c444a,
                    0x19865754348df8e4,
                    0x25bbaa51cae7a66a,
                ])),
                Felt::new(BigInteger256([
                    0xc6d4264abe919766,
                    0x38030e9fbc3f76e6,
                    0xc9906cc3585162b5,
                    0x3107623873c9dff7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0cbbe61b76668dd9,
                    0xb81703ab48f94f05,
                    0x1ef75d33a10c23ea,
                    0x6dafb7fd54a20657,
                ])),
                Felt::new(BigInteger256([
                    0xfe5623c0eabe2982,
                    0xa55ef29fcfd60192,
                    0x03ffc6533d008681,
                    0x206d560700f15c2f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8dce0c9e4be0d9cf,
                    0xf8a3fe34ac529d7d,
                    0x80afc631a133a66b,
                    0x541fbc302a180e16,
                ])),
                Felt::new(BigInteger256([
                    0xf29aa52dee6a3ebf,
                    0x4d4038e8e2568dd5,
                    0xb658fc6c5bef3ec8,
                    0x4786fab917267a75,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x89ddd80d6f192e2a,
                    0xf274d72ae9823c84,
                    0x6794654b4cf7a506,
                    0x6db1b82131e497d8,
                ])),
                Felt::new(BigInteger256([
                    0xc39c4bd6b238dbf4,
                    0x3c91b73d0a7cdeba,
                    0x91ba93e641496f1a,
                    0x70bb5abb76300809,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x35a145b86b9f631d,
                0x6b23d3b4323c71ed,
                0x15921672263a11ea,
                0x449a05cdde880be2,
            ]))],
            [Felt::new(BigInteger256([
                0x0e118e64f4ac0ae8,
                0xa52983cd0d20d777,
                0x5f831a2fc7d448d5,
                0x0d6beec9625d6b40,
            ]))],
            [Felt::new(BigInteger256([
                0x1e47a8330894fd04,
                0x23ab2f26bdc87159,
                0xba2f74325116d442,
                0x50f806bf60e0a9ff,
            ]))],
            [Felt::new(BigInteger256([
                0x4c965a786895eee6,
                0x8d631af20b9f8482,
                0xd1470b46bdccc99b,
                0x13071d70dffa6ff9,
            ]))],
            [Felt::new(BigInteger256([
                0x4df18c68f14fde26,
                0xacf7b3f9c2f3dd35,
                0x3d931104816d103d,
                0x1f8956456675cf5a,
            ]))],
            [Felt::new(BigInteger256([
                0x1edd25263632f1e1,
                0x5a1e074debee5a88,
                0x7452147bdde17277,
                0x4c4a2b8410deaba0,
            ]))],
            [Felt::new(BigInteger256([
                0x8fd6c011666cf70c,
                0xbe16ab850608a82b,
                0xc285aa38fda916b2,
                0x5d1ce957fcc84680,
            ]))],
            [Felt::new(BigInteger256([
                0x659633c9c8ca85a6,
                0x27b334fd5c43e821,
                0x2b0ce98b514f69be,
                0x57d0ec0f336607be,
            ]))],
            [Felt::new(BigInteger256([
                0xda5935e74afb5f9d,
                0xa4b8d1fdf72a23e8,
                0x5297a4c27bc1538d,
                0x27c251e25d97d414,
            ]))],
            [Felt::new(BigInteger256([
                0x52ce39ce25e022b2,
                0x6e4a4495edec5723,
                0x9dfefb5f33e594d8,
                0x72d00181285902a2,
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
