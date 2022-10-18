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

    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger384([
                0xc907821eea40bbe8,
                0xa046d9ce60c19643,
                0x6d8927e94a8a6274,
                0x04da24f46481113c,
                0x2a1e81676e241192,
                0x015ff80837b4587f,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x3d552a9b7387493e,
                    0xfebc15bfb7700e15,
                    0xa1487a0ae81795bd,
                    0x42263e321336e835,
                    0x4351fd39b91cb755,
                    0x00319349d31f1c3a,
                ])),
                Felt::new(BigInteger384([
                    0x71e34b32ebe36ed7,
                    0x6080757c0a6865e1,
                    0x913aae7f970b14d4,
                    0x50e8fa34a0f57874,
                    0xbfe5e51b7245bc95,
                    0x008ac319873e09cb,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x758c9aa50b6302fa,
                    0xf3fcb5fa8a1ba32d,
                    0xa46868daa748c7bc,
                    0x39373b1fd3b376e2,
                    0x1bcf531846a158fe,
                    0x016b5e0fea70c0cf,
                ])),
                Felt::new(BigInteger384([
                    0xd765d0ed76930e10,
                    0xdfe3886bca964f15,
                    0x95244022ad1176f7,
                    0xb19d92784f6b9ea6,
                    0x3ea83f4fb00d4b4a,
                    0x009272ca2cf506a7,
                ])),
                Felt::new(BigInteger384([
                    0x4108bf722cc5e22c,
                    0x98c096ce664dc218,
                    0x807de6c8465c9b26,
                    0xb64c9a21eb24adb7,
                    0x3d22f4bead56fd1d,
                    0x00f8965b0e63d8ce,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd1c2e2fb966680a7,
                    0xa9adea3ab62fe080,
                    0xbd9adb276cd48c71,
                    0x8d7a3753e3dcff42,
                    0x71ff088dba34023e,
                    0x0177df429219fcf4,
                ])),
                Felt::new(BigInteger384([
                    0xe370c35a480e9c3d,
                    0xebdfaf5e1486f263,
                    0x0beef61b4fb5e474,
                    0x2f8e257a6eb1c8cb,
                    0x28b33b4d1df7886a,
                    0x00019a5ab31fa9c4,
                ])),
                Felt::new(BigInteger384([
                    0xc99b5deacdae7d11,
                    0xc0aed99fea67edf7,
                    0x83958470a015542b,
                    0x18afca84b29ea936,
                    0xf5a4c8f0a06eaab0,
                    0x018bf5f78a118e71,
                ])),
                Felt::new(BigInteger384([
                    0xf41d70352b7d43fd,
                    0x50d46b82aa8f9afb,
                    0x029c77a58c78e410,
                    0x4fc3202610af5fcb,
                    0x7b0e8432a101e01c,
                    0x0138c3c02832ce3d,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x41dcdb2ceb005b4e,
                    0x7678e56e66c5672e,
                    0x6eb47711246c61c4,
                    0xce45bb8ff33ed24f,
                    0xc6fe0267b48e2c3a,
                    0x0103ad6f766d8107,
                ])),
                Felt::new(BigInteger384([
                    0xb0ccf3975959d000,
                    0xcb5d1c1f71b45aa7,
                    0x21351f90dd4027ca,
                    0x6f35e754cf2ffb15,
                    0xdba994f231789f8b,
                    0x0026a6f92752da29,
                ])),
                Felt::new(BigInteger384([
                    0x56ca0dbe270d9bdc,
                    0xd0c00765a8b823a9,
                    0x1073e570ca8ae851,
                    0x3dbca3029685335c,
                    0x5ebbe273086ab1ba,
                    0x019c6c39d041adb1,
                ])),
                Felt::new(BigInteger384([
                    0x5b86bc8d4663f211,
                    0xaf7bb43ca863491b,
                    0x9f3d2f6e005a6200,
                    0xd5a067718c46b39e,
                    0x0441000a36e938d8,
                    0x0121bf2605682d42,
                ])),
                Felt::new(BigInteger384([
                    0x37ae3d52bdf879a1,
                    0x154920ec4a6b32bf,
                    0x16dc273c0e7e9221,
                    0x2b8877cb85f93554,
                    0x25ce07f94c5cebe2,
                    0x00971c8a9e1e3ea6,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe35e410c789fde33,
                    0x07b8f2e3d90952df,
                    0x646b6cfc67494481,
                    0xdcd77475fc69aba7,
                    0x835d907fec0e759f,
                    0x0119d4005aa73e33,
                ])),
                Felt::new(BigInteger384([
                    0x765679088b58c6c1,
                    0x59d66a952f69c317,
                    0xe387b4e5528823ce,
                    0x2ed70f92746a9989,
                    0xe6a48151e8670589,
                    0x00533179f574dab5,
                ])),
                Felt::new(BigInteger384([
                    0x1c6d46a94cac402a,
                    0xadcf662454f516fd,
                    0x6e57bacf587cdb75,
                    0x76e7a8b5ee36cfe6,
                    0x4f92f8fa85d09bf6,
                    0x0130681aa96f4e49,
                ])),
                Felt::new(BigInteger384([
                    0x0ecb58b94c9c5839,
                    0xc62a6c1d6a1416e9,
                    0x84f2a10d075e9a25,
                    0x42ee1f781dd395b0,
                    0x2a3a5644119b3bb7,
                    0x01919b404370c35c,
                ])),
                Felt::new(BigInteger384([
                    0x7b9d362830580e99,
                    0xeee4774b04121656,
                    0xee5f62ef71ca6257,
                    0x089dd8b08f256e0a,
                    0x177db25570dcf11b,
                    0x017bc8f3bdc94bf9,
                ])),
                Felt::new(BigInteger384([
                    0x1290cb35cce7e63e,
                    0x485de7303212b1d7,
                    0xc78f7e038e8972e0,
                    0xd9f1e5e2b12dbfe5,
                    0x5629077ffd221680,
                    0x005a2d3fbf1735bd,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xc9a3aa691130b53d,
                0x724fea0b77ec5922,
                0x3d7d3c7a710d2999,
                0x8206c4c2002264af,
                0x6b7ec6869a94ce4b,
                0x018ec80a7a131db3,
            ]))],
            [Felt::new(BigInteger384([
                0xd6eb66356d6c34d8,
                0xffec1d3fe30a9b98,
                0xc8e695c9ebb483a7,
                0x486c36a21915d931,
                0xd5c3113f1ba1da79,
                0x00f4460b3812a0b0,
            ]))],
            [Felt::new(BigInteger384([
                0x743b8dc1affae49e,
                0xdb3a80fcf0bbf8a4,
                0xa76c747c243e3024,
                0x80a992902cd3350f,
                0x5e5331ae9401b486,
                0x0076043c794811c0,
            ]))],
            [Felt::new(BigInteger384([
                0x64cdfff9e51942c0,
                0x69abf6faa77e83b5,
                0x3abd651e3ae26abb,
                0x1b98799fe8e16235,
                0xe5e52d29a039d046,
                0x00965cc85a0268c0,
            ]))],
            [Felt::new(BigInteger384([
                0x32342bb3ddd28b05,
                0xa1c2b83df83142fa,
                0x44fac4691cf7d4ce,
                0x7c9fb93cb264868d,
                0xb026c6e789ad1569,
                0x000825311dc29668,
            ]))],
            [Felt::new(BigInteger384([
                0xb1edc25f1b1a76fe,
                0x556fd11fa2428997,
                0x5be0f5148749a77a,
                0xbba8830c49296aef,
                0x422f66ebb96428e2,
                0x001274f5487dcdac,
            ]))],
            [Felt::new(BigInteger384([
                0x5ac8410fc5e6f6fd,
                0x6b4e53ea507a646e,
                0x5568801a3241293d,
                0x6a031308ecc4c34a,
                0x8a1fc49c96e8cf62,
                0x01a3410a537a7968,
            ]))],
            [Felt::new(BigInteger384([
                0x83fa51496d3076d3,
                0xef8a53d0f4f7c216,
                0x017cb1af23a0a7e2,
                0x42c9d9b774f341ca,
                0xfbe87273aacbd594,
                0x001436674a4c3d9b,
            ]))],
            [Felt::new(BigInteger384([
                0xf8ca5c01c3f8a912,
                0x487011044753d0b7,
                0x6ef7066b08470d0a,
                0x28650824153d9856,
                0xcabf021d458bdede,
                0x0156ebfe965ebb4d,
            ]))],
            [Felt::new(BigInteger384([
                0x050085faf0ed344e,
                0xabafd105fca2678d,
                0xab86b99ce5b38c0a,
                0x1b0362fe1e444091,
                0xac356c4c581db647,
                0x0087ab0491de4587,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
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
                Felt::new(BigInteger384([
                    0xde6d95c2edacb809,
                    0x8017fd28783396eb,
                    0x5a7e25736d0620a0,
                    0x726a83ae82d96d48,
                    0x908c215fcd24b766,
                    0x0177bd5777930cda,
                ])),
                Felt::new(BigInteger384([
                    0xa7f7b1b283174a46,
                    0x60614ace621cc05b,
                    0x2d97a763c8a15433,
                    0x4a9eb3368257d3ac,
                    0xbcdffe1d8e556912,
                    0x00f3205803843fd3,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xfb4334b600c2dc68,
                    0x4c5e0ad846c5f1a2,
                    0x2af9a9350b75c5fb,
                    0x1ef1b0755ef3ae52,
                    0xfd1af0413eb5cef9,
                    0x013fdd446f6d0182,
                ])),
                Felt::new(BigInteger384([
                    0x2a198caca9a8f1ef,
                    0xc8d349e260bbef34,
                    0x0fe830451b92eba8,
                    0xbaf714033698fd90,
                    0x28506424c69b4271,
                    0x0080da2696c1d3de,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa4f438de64e5d1c0,
                    0x0a9a5578bbed0088,
                    0xfef938d0f73b4801,
                    0x7bbf06ea2ebc3b4d,
                    0xca46e120b0adb535,
                    0x0049b0c015c094c6,
                ])),
                Felt::new(BigInteger384([
                    0x69dc993eb293dea8,
                    0xb1399e5924f86fc1,
                    0xca193248c8b01ccd,
                    0xe5f7c164bb32e163,
                    0x6f94f54a9878fe6e,
                    0x012b64f74b582fc8,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xdd47336a8a10fff1,
                    0x2ee79a3ed7ae5ff8,
                    0x110bc75d78f5d423,
                    0x93322f5c0b56080f,
                    0x4631d1e241086bb3,
                    0x004a4cf8ff833f09,
                ])),
                Felt::new(BigInteger384([
                    0x2aa9fc048da78057,
                    0xe874685718ae3eeb,
                    0x2b6f1be03f3bb88f,
                    0xb591ed52ab4291ee,
                    0xd1008dda5ea190b2,
                    0x0005d4cfd0b06cb0,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x21ee553bb0051563,
                    0xa80fd0164dcf430f,
                    0x20a129b00d408a8b,
                    0x4129221a0be8dd10,
                    0x046d24f3d43a42fd,
                    0x0107b3f4a231af2c,
                ])),
                Felt::new(BigInteger384([
                    0xc1550230679008af,
                    0x2d8a98200d346d31,
                    0x3391b4eeba67eee1,
                    0x64577d850f97d8d1,
                    0xc048e674ad51fb24,
                    0x0001be6021cdcb57,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb90a3546f6704d42,
                    0xdeec09216d48e699,
                    0x9b5e6629ad830db1,
                    0xfd23b68c445c6c9f,
                    0xf154accf8987964d,
                    0x0106ef7644dde996,
                ])),
                Felt::new(BigInteger384([
                    0x00d8a947ea67e6d0,
                    0xf1350b9e2e58b346,
                    0x67705aa61c514252,
                    0xbc987a9c8444a07f,
                    0xb17d8235c9221722,
                    0x00cd1aca40f83925,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xd54fe9d8c0680953,
                0x4913f40be66d5ca7,
                0x43d347bed99f1639,
                0x9e7bda182cdb06be,
                0xc07a12657051903f,
                0x0093608ff494e21d,
            ]))],
            [Felt::new(BigInteger384([
                0xdc763f54be46fd5b,
                0xa0055a1a7af41a25,
                0x7c7ae2432ddbddad,
                0xd468df702617373e,
                0x4435f363c4dc443e,
                0x00aa017fd5568759,
            ]))],
            [Felt::new(BigInteger384([
                0x0ff9ca2c99002264,
                0x71f3fafbd3946547,
                0x543091c14e463361,
                0xb53f2f90670621a5,
                0x15a9499cb79bf5ba,
                0x0033ae13d76cca8e,
            ]))],
            [Felt::new(BigInteger384([
                0x2f1cbaaa6e1b3b81,
                0x2de6602bc61b9fa6,
                0x01241fc5cb6cfab2,
                0xa5b64c3299976170,
                0xac9bb265dba610a0,
                0x01950f59ee6a4763,
            ]))],
            [Felt::new(BigInteger384([
                0xe6e08c266992efa4,
                0xa2fb771bced269dd,
                0x8f0d5f2240507e94,
                0x6064fa4e64fd7599,
                0x253008989fdedd46,
                0x015c4b2d1f7b3a9d,
            ]))],
            [Felt::new(BigInteger384([
                0x54f868793dae4eb0,
                0x32cf7524650fd11b,
                0xa74029cb337eb0e7,
                0xfb070b67e70f57d0,
                0x5fa63757b79511d5,
                0x011fedcbd47efc83,
            ]))],
            [Felt::new(BigInteger384([
                0x1a25cad68bfdecbc,
                0x46952c5aab186377,
                0x2dc2bccd75202af3,
                0x8af41d87f0faf249,
                0xe35c21d80fa16260,
                0x00150bc6b52b7080,
            ]))],
            [Felt::new(BigInteger384([
                0xdf52530fac116976,
                0x6db431a7bf4ebdec,
                0x530d97f115df4855,
                0xdc2637c03a6b7a82,
                0xee80144f10e311d6,
                0x016dafb3fb0993da,
            ]))],
            [Felt::new(BigInteger384([
                0x13668df46165675a,
                0x6c5c9c291602fcd2,
                0xbf26c8ba6fee3b27,
                0x656a80b3539dc1f2,
                0x4a616004eb6848e4,
                0x018c5b48f1a6dedb,
            ]))],
            [Felt::new(BigInteger384([
                0xdd7f2752d5da5fa7,
                0x00c99e966733a4e4,
                0xfe059b3325111b95,
                0xbdb0a7eb29e43800,
                0x1466ebddb9455e85,
                0x016afd6687603fda,
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
