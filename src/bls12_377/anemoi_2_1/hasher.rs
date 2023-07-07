//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBls12_377_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiBls12_377_2_1 {
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
            AnemoiBls12_377_2_1::permutation(&mut state);
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
            AnemoiBls12_377_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiBls12_377_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiBls12_377_2_1::permutation(&mut state);

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
                0xa9cdfcae1797ccaf,
                0x7a4d889d248f869a,
                0xc1569404ea1fe9cb,
                0x13b3ea0b624e835d,
                0x2dbfc0f2d15d7700,
                0x0066c1d06bcc0167,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0xb54a6a62a86599eb,
                    0x4ef6994e55ee522f,
                    0x42d9a55366837932,
                    0xb15ae77779df338c,
                    0x7e83666c4be11a30,
                    0x01a060cd14ef4c15,
                ])),
                Felt::new(BigInteger384([
                    0x3e1f2d8379900a2f,
                    0x8924cdb1a7aced40,
                    0x75ba822d9f94608f,
                    0xb643658b5457459c,
                    0x039c532900478196,
                    0x014fd44121e3572b,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc2fe1b841cb5c3cc,
                    0xd9e6c19c28e6a343,
                    0x1ae0c9981e1d57d4,
                    0x27fc5043b5d4e62b,
                    0x780e914eae15a557,
                    0x00123f7eab401a9b,
                ])),
                Felt::new(BigInteger384([
                    0x64628b8462bcd15e,
                    0x2f2a56d718b2418a,
                    0x3b5d7f7df1c76b79,
                    0xb6cfac487c55c24d,
                    0xce0d310dd37e7ffa,
                    0x0029fed493ec6bb4,
                ])),
                Felt::new(BigInteger384([
                    0x0db9b9503fcfaff5,
                    0x76c28dd24ad187ff,
                    0xa9d4f2db70d5d7bf,
                    0xaa2488137449f7de,
                    0x7b1c34792635ebd8,
                    0x01351c1b2e777259,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x7d2ba624fa7921f9,
                    0xc84811684224f254,
                    0x363856b9e3469458,
                    0x18204789e9eed90c,
                    0x33d072000bb189d0,
                    0x0154b3896cf3c8e0,
                ])),
                Felt::new(BigInteger384([
                    0x118871da71b95459,
                    0x82a5fc359c4dff50,
                    0x3f8f8201305cf373,
                    0xe302933dbfe08a5a,
                    0xab3a3ac0e9692702,
                    0x00a6b4e137ce028a,
                ])),
                Felt::new(BigInteger384([
                    0xbef8b0c928cc6220,
                    0xb8026884179e0088,
                    0xf196445ea500784e,
                    0xf2def7bf43f14240,
                    0x304e540fa60514e7,
                    0x0085fe5b783970cf,
                ])),
                Felt::new(BigInteger384([
                    0x54717e173037154a,
                    0x396c7a849e3f7529,
                    0x297abd49bd13e015,
                    0x7a595f99c0f18d8c,
                    0x45a6c45dfcce3382,
                    0x0123882a32bec3e3,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa7a1595910720d6e,
                    0x98a142db4856bca8,
                    0xbfd696bd3e92f8f1,
                    0x1e7386d53708d027,
                    0x9fbf2c4c86755e90,
                    0x007aa8c79b6de521,
                ])),
                Felt::new(BigInteger384([
                    0x34afc92cac010049,
                    0x87dcb66eca4d4dad,
                    0xfb5ccebea05d09a0,
                    0x5812f9fe5e10a54c,
                    0x90ac0109b2e7d9e8,
                    0x00371784f9982746,
                ])),
                Felt::new(BigInteger384([
                    0xc6b6422db422fb89,
                    0xf50fd92453f31813,
                    0x930ae597c6327ebc,
                    0xce8e099a87322763,
                    0x991dfe81c1ed98da,
                    0x011501474148f402,
                ])),
                Felt::new(BigInteger384([
                    0x0cdaf2ea6226dcc9,
                    0xd83eee9abe081700,
                    0xbe1d7d3e046636fc,
                    0x8129b642e61ebe4a,
                    0xdba0b2b72004d449,
                    0x0023e50a62aac67b,
                ])),
                Felt::new(BigInteger384([
                    0xcd3acb9f2c37b705,
                    0x5d72855fc96d91c4,
                    0x71e6f2adb0680c04,
                    0xf2a43f72bc2b119c,
                    0x6304cfae9c170ccf,
                    0x016f2d94d4549226,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf1ef166b5b80224a,
                    0xfd7bb7d7780329a0,
                    0x2743ad066e31041f,
                    0xdf9000582dc71046,
                    0xef0df3689851755a,
                    0x001c8656163607d4,
                ])),
                Felt::new(BigInteger384([
                    0xac6af5159f6324b4,
                    0xd3e5274c0d48bbce,
                    0x078d46668c60ddb3,
                    0xe6c0bcd281ada3dc,
                    0x1b6dd6e053276ba9,
                    0x00601a5591f5aab1,
                ])),
                Felt::new(BigInteger384([
                    0xb6ae96d28a0e33c6,
                    0x6ec27f05238ffa23,
                    0xbd3f08f8bfc6794b,
                    0x2d2ed2eeaec0b494,
                    0x8eb202e7c224a518,
                    0x008f9eab5a09c559,
                ])),
                Felt::new(BigInteger384([
                    0x6c3c037316579fec,
                    0x26ec56ca5a63030e,
                    0x71eb31b9f29de2b3,
                    0xc08fbd42e2b05a3b,
                    0x58c98b8eaa538182,
                    0x0001c28a0ac243c7,
                ])),
                Felt::new(BigInteger384([
                    0x4ad06d09a5b7eae3,
                    0x01b16f512bc1b424,
                    0x386dfca1d6c5408d,
                    0x5c468d773c54165f,
                    0x01f121f3d30f7fe8,
                    0x009c28fdccad1a56,
                ])),
                Felt::new(BigInteger384([
                    0x1e59cd144765ed42,
                    0xd205c93ab6619d7c,
                    0xf772f6e8992c5cb9,
                    0x177f2b7cdd61f03c,
                    0xf07cbad88ee4995e,
                    0x005d776b5f467c91,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x73e883d74459b235,
                0xcd6aa4033c3ec158,
                0x16e2e3c876891aaa,
                0x49743f0b2d3cb5e4,
                0xb21f90ac51272745,
                0x0112069d31309c1f,
            ]))],
            [Felt::new(BigInteger384([
                0x3f0894a0819f87f2,
                0xd33fb49671d59d74,
                0x850967d5d8fd15d2,
                0xa026551a6ae673d4,
                0xf36dfff3dc765198,
                0x002095c44a1b7063,
            ]))],
            [Felt::new(BigInteger384([
                0x106da0f07f6e841e,
                0xa6b81eae3cd1d3b0,
                0x59f1863fe7411a9e,
                0x57fac7f7846346c8,
                0x10c97e173217d3c2,
                0x0098ab0d735a8962,
            ]))],
            [Felt::new(BigInteger384([
                0x6a7944cd235c376d,
                0x11159372ab0bbed4,
                0x454523abf6442fd5,
                0x98c05bd247b5250d,
                0xebeaeeb18859b80e,
                0x01a467be0db7e73c,
            ]))],
            [Felt::new(BigInteger384([
                0x3a3af82215d5b915,
                0xb62155dacfdcabe5,
                0x2bee452a44176774,
                0xdb234e411a09c7a9,
                0xaa40ed928b0541c8,
                0x012fa517814cde4c,
            ]))],
            [Felt::new(BigInteger384([
                0x2550caea08e694ef,
                0x07a967dfd8262aa7,
                0xe374d7bad3f5c9ff,
                0x8ac990cc6aade4aa,
                0xd56d77bd2237d162,
                0x00527e9588963e3d,
            ]))],
            [Felt::new(BigInteger384([
                0x759c5f586e34ee65,
                0x358260cbd58888e7,
                0x1972f125822153a5,
                0xcf0118ec1ae0c317,
                0x74e3b36b8df9966d,
                0x0069896563a2bbb4,
            ]))],
            [Felt::new(BigInteger384([
                0xcc09642ac1a9bc92,
                0x7fab091458f0b27d,
                0xb5849c257ed5ddc1,
                0xaa832b651621be1d,
                0xa9dbcd23c027067d,
                0x0184da1ef914470d,
            ]))],
            [Felt::new(BigInteger384([
                0xd07f2066576dd402,
                0xe732618e5906f114,
                0xa3463e97abb76fe3,
                0xfd405cf3be914bcb,
                0x34bbdfa902dedbab,
                0x00e4459acae174cf,
            ]))],
            [Felt::new(BigInteger384([
                0xc194acbc850f079d,
                0x80ab670b2470bf83,
                0x593d858123f6d40e,
                0x635bed956aed2934,
                0xb0f8aadb45bb516c,
                0x00472c2909280645,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_377_2_1::hash_field(input).to_elements()
            );
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
                0x73e883d74459b235,
                0xcd6aa4033c3ec158,
                0x16e2e3c876891aaa,
                0x49743f0b2d3cb5e4,
                0xb21f90ac51272745,
                0x0112069d31309c1f,
            ]))],
            [Felt::new(BigInteger384([
                0x3f0894a0819f87f2,
                0xd33fb49671d59d74,
                0x850967d5d8fd15d2,
                0xa026551a6ae673d4,
                0xf36dfff3dc765198,
                0x002095c44a1b7063,
            ]))],
            [Felt::new(BigInteger384([
                0x106da0f07f6e841e,
                0xa6b81eae3cd1d3b0,
                0x59f1863fe7411a9e,
                0x57fac7f7846346c8,
                0x10c97e173217d3c2,
                0x0098ab0d735a8962,
            ]))],
            [Felt::new(BigInteger384([
                0x6a7944cd235c376d,
                0x11159372ab0bbed4,
                0x454523abf6442fd5,
                0x98c05bd247b5250d,
                0xebeaeeb18859b80e,
                0x01a467be0db7e73c,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 94];
            bytes[0..47].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..47]);
            bytes[47..94].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..47]);

            assert_eq!(expected, AnemoiBls12_377_2_1::hash(&bytes).to_elements());
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
                0xa1c71a5395c36b70,
                0x5a06f597cac1985d,
                0x66126e92929c5b9e,
                0x253f52ac010f806a,
                0x581db69852589ebf,
                0x00dd9b8d65e15c2f,
            ]))],
            [Felt::new(BigInteger384([
                0x937b67c3f43578c2,
                0x6d82843a0f4c7a5e,
                0xca672802bd9eace8,
                0xb494c49f6d26b6ab,
                0xe549757d5902535d,
                0x012ce030677330f5,
            ]))],
            [Felt::new(BigInteger384([
                0xce97990856479af7,
                0xdc9d6d61c74b7069,
                0x3106841033e92a8c,
                0x8f0ee4cfed3f4ed7,
                0x53291f6fe82371c8,
                0x0143f0533d41809f,
            ]))],
            [Felt::new(BigInteger384([
                0xce543ba668a8131c,
                0x960e0f09354d3c7e,
                0x36ef29dcef165c01,
                0xf02aa3e06ea78753,
                0xdf94e6ac3b6e4d62,
                0x0017e3353986f2c8,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_377_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
