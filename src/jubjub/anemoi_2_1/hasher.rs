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
                0x03eca0095feb34f1,
                0xf8278449bd8b7b4c,
                0xed922c0eef68a4ab,
                0x1f9a01059573ec28,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x7255955c0a2844e2,
                    0x20faae79fbf6899f,
                    0xc51004ff47bcf76b,
                    0x309af2a383e44090,
                ])),
                Felt::new(BigInteger256([
                    0x6ad7088065f1b7ec,
                    0x922786acfa452708,
                    0xb231d8e8d7a83301,
                    0x52269fbc1de5d75a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x998f89bb4afe6430,
                    0x61fd39ab311eee60,
                    0x9961254626bf0b79,
                    0x5980e330bd8f451d,
                ])),
                Felt::new(BigInteger256([
                    0xe14fc94358c31c7f,
                    0x57f883c03561a859,
                    0x4ed2262365399e16,
                    0x0e6bec83912ce1a5,
                ])),
                Felt::new(BigInteger256([
                    0x2fce478d094cbd9b,
                    0xe535d2ca881b08f0,
                    0x9949800cb20cae3c,
                    0x58e391fa9917d011,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe76f83c643f8cac4,
                    0x0a9da165516faa1b,
                    0x04193bc63a900352,
                    0x2d99e219eaefaf4f,
                ])),
                Felt::new(BigInteger256([
                    0x59a6ce85deb78dee,
                    0x371131883a19d45f,
                    0xfd2cc142d0e684c0,
                    0x3fb03a3243c8301a,
                ])),
                Felt::new(BigInteger256([
                    0xcaa56a6b730913f0,
                    0xcbfde89627eb65a2,
                    0xfec1c51483996c1d,
                    0x367673b5a0c81f11,
                ])),
                Felt::new(BigInteger256([
                    0x1f7beabf3467167c,
                    0x594d4fd0c501e3fe,
                    0x36e774fe56df7d13,
                    0x09c56676fb2dd566,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x64c34f3f8928f66b,
                    0x74da22e6260fef6f,
                    0xf099771a1c8664c9,
                    0x431774f603c88f69,
                ])),
                Felt::new(BigInteger256([
                    0x03c8e250d499007d,
                    0xb54021340d11b7fc,
                    0x433092dd7e64712f,
                    0x58a658870d0ac9ac,
                ])),
                Felt::new(BigInteger256([
                    0xbc6496077cee7683,
                    0x511e3667d16f21ac,
                    0xbb2b72a6ccc0c3c2,
                    0x0f244ef95dc1bc6b,
                ])),
                Felt::new(BigInteger256([
                    0x5306ad4db87330fd,
                    0x814c5f683412ea83,
                    0xdfdf87a1ae4d6594,
                    0x4ec810c957c87a4f,
                ])),
                Felt::new(BigInteger256([
                    0x74bcf9d0df31106f,
                    0xf01ed2395cbde0c8,
                    0xf9c09088819e7208,
                    0x3bf2bb4c8a45321e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfe11e291ca3e6b59,
                    0xeac71d3480e5c58d,
                    0x08b9e1496ef0c48e,
                    0x2309c028e1cd309a,
                ])),
                Felt::new(BigInteger256([
                    0x3b1a10caadbb0172,
                    0xcdf59231a3212cc0,
                    0xd295becb0c01a9ba,
                    0x62a4c7c63ed96e78,
                ])),
                Felt::new(BigInteger256([
                    0x253739d91e084521,
                    0x14f4f0c1fe1688c7,
                    0xdf5278d8ac9852c7,
                    0x2965a8e141d4ce72,
                ])),
                Felt::new(BigInteger256([
                    0xbe803aefb00f3000,
                    0x1aeb92bd71080740,
                    0x30eaffa189a12580,
                    0x317f080e81bc7608,
                ])),
                Felt::new(BigInteger256([
                    0x767fccd27f68d9e2,
                    0xe8e64194b9c2854c,
                    0x2cd4151cc9dfa870,
                    0x7035988a219aacaf,
                ])),
                Felt::new(BigInteger256([
                    0x0a1d9a346fb7f9f3,
                    0x33a4ef56740df0a1,
                    0x028f903d989ca737,
                    0x7271eb280cabe551,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xc614ddd54021a538,
                0xc45d1e979222b59e,
                0xda5864704282cf7a,
                0x44e448baca30eb29,
            ]))],
            [Felt::new(BigInteger256([
                0xa09396c86db64676,
                0x5b095db2ce0685ea,
                0xed8b72bce995b7b1,
                0x08397c5f342e282d,
            ]))],
            [Felt::new(BigInteger256([
                0x9715d2a4cdf12e6d,
                0xf324f73f3b46d4b0,
                0x4ab2331022eb9e04,
                0x4da6771da91cc574,
            ]))],
            [Felt::new(BigInteger256([
                0xd4b75906d4ebc690,
                0x4766eaccd268daee,
                0xdc95e0a8c78b9792,
                0x2fd05d66c1c462a6,
            ]))],
            [Felt::new(BigInteger256([
                0x68a58daf97d457b9,
                0x5637e4abf0c89a03,
                0x489ee42fe25f8e2b,
                0x1c93c80b83053554,
            ]))],
            [Felt::new(BigInteger256([
                0x74acd0e554f62f28,
                0x91517013baf9efc8,
                0x6df0cdc8f8b7af12,
                0x0c35252216e8843e,
            ]))],
            [Felt::new(BigInteger256([
                0x76459e4546991bad,
                0x76adffe40e4c5574,
                0x9956dc23ab1ed917,
                0x412084263c29b422,
            ]))],
            [Felt::new(BigInteger256([
                0x9a2222d455b9d70e,
                0x8dfc55cccfb6bbf5,
                0xbea5beb185f3f5d1,
                0x27193d54f5da2796,
            ]))],
            [Felt::new(BigInteger256([
                0x5d3f8039f2855cc9,
                0x238f15bf31c4ff3d,
                0xe333734e26ae2bff,
                0x59488221512dda7c,
            ]))],
            [Felt::new(BigInteger256([
                0x4284b8bb6e5ed8f4,
                0x9a09ba1861723765,
                0x4a5c110be7179b16,
                0x6c01dd66c6918ac1,
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
                0xc614ddd54021a538,
                0xc45d1e979222b59e,
                0xda5864704282cf7a,
                0x44e448baca30eb29,
            ]))],
            [Felt::new(BigInteger256([
                0xa09396c86db64676,
                0x5b095db2ce0685ea,
                0xed8b72bce995b7b1,
                0x08397c5f342e282d,
            ]))],
            [Felt::new(BigInteger256([
                0x9715d2a4cdf12e6d,
                0xf324f73f3b46d4b0,
                0x4ab2331022eb9e04,
                0x4da6771da91cc574,
            ]))],
            [Felt::new(BigInteger256([
                0xd4b75906d4ebc690,
                0x4766eaccd268daee,
                0xdc95e0a8c78b9792,
                0x2fd05d66c1c462a6,
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
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xa28002b38097d0ef,
                0x37e23b24e9947e4b,
                0xe9874f1bc411059a,
                0x05c9d1c3dfb803eb,
            ]))],
            [Felt::new(BigInteger256([
                0xc973bd76b8521227,
                0xf4217294db16c2b2,
                0x3bc0665b501eb299,
                0x637eb5a706c38402,
            ]))],
            [Felt::new(BigInteger256([
                0x72318de12f13d23a,
                0x42e3304cdcb13d28,
                0xc8def1b363aa59cd,
                0x28a507048f4c207f,
            ]))],
            [Felt::new(BigInteger256([
                0x1746cfef08a48cc7,
                0x3198e2542e047983,
                0x788906f2f49748e5,
                0x4837f31c063a5fee,
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
