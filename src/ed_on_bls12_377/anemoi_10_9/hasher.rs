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
                0xc737eb02b8134351,
                0xe6748f858e6a29f4,
                0x4d3e0a26b5b97f4b,
                0x102a127fddd71188,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x878545d0fb91d0a5,
                    0x03c9eb1ca91fe896,
                    0x9a3c8c45557ecb26,
                    0x058539e7a9b9e002,
                ])),
                Felt::new(BigInteger256([
                    0x2c837588cf81d69e,
                    0x243d3139b544c6b4,
                    0x0e11cf26aa41e84d,
                    0x0a1bfd0a3b60ba35,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1b8548caff219040,
                    0x4c6e3efad967bda8,
                    0x9ba4424955b40a81,
                    0x0322a076eddf7234,
                ])),
                Felt::new(BigInteger256([
                    0x0195c662195294f5,
                    0x00aa9b07c6b8928b,
                    0x0e03f73c5379f233,
                    0x11ac3a5992d25f2d,
                ])),
                Felt::new(BigInteger256([
                    0xc4a006e0418ee353,
                    0x8d005fa8bb79ac09,
                    0x2236eeff06993e89,
                    0x10e9a0b40ea9d149,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb790b1dbd641a269,
                    0xefb914108e66cb9b,
                    0x6d002a614ccf2cb4,
                    0x0c8bdfb810eccaf2,
                ])),
                Felt::new(BigInteger256([
                    0xd33680eeec8b8ae5,
                    0x31860d76ee8e0e1f,
                    0x4dd7650c84524860,
                    0x0783db3611016070,
                ])),
                Felt::new(BigInteger256([
                    0x5980268af72c0742,
                    0xb796598ec66ce73f,
                    0x785dd8587ba0037c,
                    0x069911521731c6d0,
                ])),
                Felt::new(BigInteger256([
                    0x35d4f64247bbe6b5,
                    0xcea5078421334fe3,
                    0x4dc28b7e8861763a,
                    0x08aba548714013f4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x66bbda9bdb02f6dc,
                    0x2c80dcf0276ed4e6,
                    0xea8f9a0080c8130d,
                    0x0e7d04f4ee28afd0,
                ])),
                Felt::new(BigInteger256([
                    0x1719c99d272676e8,
                    0xe85ddbc9ee107d71,
                    0xca7ee3262eaa70c0,
                    0x0092d74d18c2c412,
                ])),
                Felt::new(BigInteger256([
                    0xcb435cc2f2d596b0,
                    0xcd751e7e20cd54c4,
                    0xcfcc4617b5e0238b,
                    0x073d8d19c8806418,
                ])),
                Felt::new(BigInteger256([
                    0xb14488b44a2a9bf3,
                    0x174e89648115a731,
                    0x8db87cf0045d49dc,
                    0x0d3e7503ee08d1b2,
                ])),
                Felt::new(BigInteger256([
                    0xcf2895cbd44223a6,
                    0x30bdf69e3786b876,
                    0x2679dcac36f1af7d,
                    0x017e63f7209aedfd,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0d2d90a65cc10fc1,
                    0x87c1beeccbf4f435,
                    0xcd92bfca39b0f201,
                    0x0596183fdbcdac86,
                ])),
                Felt::new(BigInteger256([
                    0x87e04eda59628f05,
                    0xb00fd0ccd1f80f91,
                    0xe8d78dd431b27d39,
                    0x0e8d41db8e4902f5,
                ])),
                Felt::new(BigInteger256([
                    0x6bdbdf344cc4b3c6,
                    0xd43e0beb7eac9c01,
                    0xbf02fcab285b7182,
                    0x0603caddea06c6e7,
                ])),
                Felt::new(BigInteger256([
                    0x39bea7a15a3209b9,
                    0x2ca5433e45f62a6a,
                    0xbdb21e49ba9ffb73,
                    0x05bd62baad311bfd,
                ])),
                Felt::new(BigInteger256([
                    0xba7826f44c645dde,
                    0xeb835e25bc31d9c4,
                    0xf671989f6762bdc4,
                    0x11772df799c15fad,
                ])),
                Felt::new(BigInteger256([
                    0xc236a084431820fc,
                    0x8d0ec48b4f4ab942,
                    0xabe8d9a106deb64d,
                    0x0bd9f8ab37d29c44,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x58df55c419be365f,
                0xb150f07c5b42fc73,
                0x11fa0040d7562b81,
                0x00e7181e3ec4cbdd,
            ]))],
            [Felt::new(BigInteger256([
                0x393f72ede0dc311f,
                0x7c5d59ccbcb67fb7,
                0xc8d60173b06c1fd0,
                0x02ee3385288a2f86,
            ]))],
            [Felt::new(BigInteger256([
                0xca26edfb634a7cb3,
                0xf1c72376e1a100d9,
                0xc0b397c60596cc6a,
                0x0b7a001b4a10a2bf,
            ]))],
            [Felt::new(BigInteger256([
                0x5b1c19f4f1f7973c,
                0x1cc7ea0d1be64be1,
                0x16c05bd761176e3c,
                0x11ac1422c3a1cbaf,
            ]))],
            [Felt::new(BigInteger256([
                0xe5374f53e0bdba65,
                0x094f932299337dc0,
                0x175b16b7d0595877,
                0x0768cb6ee702b789,
            ]))],
            [Felt::new(BigInteger256([
                0x205081855154fa55,
                0x5756a9a55125e81c,
                0x6b08c08237d7656f,
                0x05a82c8fc298743c,
            ]))],
            [Felt::new(BigInteger256([
                0xc82ed4936337e02b,
                0x90b1187ce4d3d628,
                0x9e6f7ab441bc5fe8,
                0x0be0d3c99cc2307c,
            ]))],
            [Felt::new(BigInteger256([
                0xfabdf8c1ff839473,
                0x0af31d8f264c14ed,
                0x8be4d840b9cf0838,
                0x043d7c3e4e848de8,
            ]))],
            [Felt::new(BigInteger256([
                0xdaa9b0cc82be7180,
                0xa49ad65116b64f3e,
                0x4575943f507a3190,
                0x0c402045eb52f9fb,
            ]))],
            [Felt::new(BigInteger256([
                0x7cd345842c3445b4,
                0x6e4f52940a03af32,
                0xaaf8afa61cca83ef,
                0x1214986f53baf27a,
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
                    0xc64fb09da4675c7f,
                    0x529feb42db777de6,
                    0xcb41e4eae64c36c4,
                    0x0ff257ad4222a882,
                ])),
                Felt::new(BigInteger256([
                    0xfbd2b73b98883f9d,
                    0x967931cd370e4317,
                    0xf10fa99a8f595c7a,
                    0x0d5e0944c5cd67df,
                ])),
                Felt::new(BigInteger256([
                    0x8d083b10ec1c483d,
                    0x8e69d046735ba1e5,
                    0x3e9da2abdfec165a,
                    0x06e012291636243c,
                ])),
                Felt::new(BigInteger256([
                    0x23ed7e5c0f9d7a05,
                    0xfdbe142c0e82acec,
                    0xccac204569947306,
                    0x0842036a278c6d0e,
                ])),
                Felt::new(BigInteger256([
                    0xdeeff122117ce75b,
                    0x268b4e757ab0bc47,
                    0xd3a4892fa3cd7c62,
                    0x02b67366740f38a4,
                ])),
                Felt::new(BigInteger256([
                    0xd6e071359651113f,
                    0x7ce9dcbf5550b617,
                    0x1e2e0424114d4d37,
                    0x0328e93c85404b10,
                ])),
                Felt::new(BigInteger256([
                    0x135a1aa2ef7fd065,
                    0x014f1c8cac5ceb49,
                    0x78e0a8f8018567a2,
                    0x03313c09b16ec6b6,
                ])),
                Felt::new(BigInteger256([
                    0x129158c85e45ad52,
                    0x32d284b1328375b4,
                    0x8b6e6b01bbd16e46,
                    0x0ccb4969e2fe90d4,
                ])),
                Felt::new(BigInteger256([
                    0x0e61e446a4a34ff7,
                    0x723a3d03ac7177b7,
                    0x79c7f17dbe57db3f,
                    0x1227fcdf776c95d2,
                ])),
                Felt::new(BigInteger256([
                    0xa05d93c668d45e0e,
                    0x56351da32bd27ada,
                    0x844f42d87539cf0a,
                    0x0ae7f8a38f6da882,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb394ab1d71c0f940,
                    0x775768a00c30c13d,
                    0xd41ebf8db6e45808,
                    0x0d404bd4b29b73f6,
                ])),
                Felt::new(BigInteger256([
                    0xc8ada1db8245e55a,
                    0x59ec96b8a3eaf592,
                    0x6a8d9ad1a547dc6e,
                    0x050f1be78c78a651,
                ])),
                Felt::new(BigInteger256([
                    0xce83408dd1ac5ec3,
                    0x54abf49504ca3a3d,
                    0x9999fe3a11b5e3f4,
                    0x0a952d8848419cb3,
                ])),
                Felt::new(BigInteger256([
                    0xec4170b766594bea,
                    0xd055253d3b0a5fa2,
                    0x93cecbf7116d2d97,
                    0x0b7850095cf2db86,
                ])),
                Felt::new(BigInteger256([
                    0x9759a8aa88e0da3c,
                    0xff4a379f396b1552,
                    0xac2248530610e662,
                    0x09ffff28175e9b47,
                ])),
                Felt::new(BigInteger256([
                    0x91a23f3b09153089,
                    0x1b15eee91ee701d2,
                    0xa45ebb16444d1bcb,
                    0x100b2235bc13d232,
                ])),
                Felt::new(BigInteger256([
                    0xed8136039ebcb8b8,
                    0x0c05e20c9e8d9546,
                    0x22f489b68091c388,
                    0x0a40327e71de9f8c,
                ])),
                Felt::new(BigInteger256([
                    0xbf2ef829eb42993a,
                    0x412806f87854e1c3,
                    0xdf56a5e2b26828d2,
                    0x0c75e337d03a698c,
                ])),
                Felt::new(BigInteger256([
                    0x1818fa9af25fe77e,
                    0x16d9a5e94ee8b672,
                    0xcdcf8e091d3a6a7c,
                    0x014e150eaa29fe16,
                ])),
                Felt::new(BigInteger256([
                    0xca4a2f37198e5ad1,
                    0xc228437786dccc8e,
                    0x5466e62da1138d5c,
                    0x06c2ce6a358620b7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x15ff0221c3d48322,
                    0x822591ca3fa02fd4,
                    0x334a64c14b3b6948,
                    0x0d0b0de2c989506e,
                ])),
                Felt::new(BigInteger256([
                    0xe2a525bc69e95e6c,
                    0x677a573857ebfaf0,
                    0xa2822604e2a9feab,
                    0x0d7ada96ff5fd48b,
                ])),
                Felt::new(BigInteger256([
                    0x7f37aa37c804c672,
                    0xf29106811afe3b02,
                    0x05b616ffe6e8ac4e,
                    0x036a1c0c798ffb99,
                ])),
                Felt::new(BigInteger256([
                    0x10fdf96f336661eb,
                    0x8f676c7dc5e0dbd6,
                    0x156518897db3bc18,
                    0x0658fc74887d0351,
                ])),
                Felt::new(BigInteger256([
                    0xa2c90f74ff9c0dff,
                    0xe8c03ee1784486b8,
                    0x7baefb466a042ba7,
                    0x04e1bdf76f7aaded,
                ])),
                Felt::new(BigInteger256([
                    0xd68017948a07c617,
                    0x2cc87d250afe8996,
                    0x204898dcd3b4810c,
                    0x11090843d5aa16e9,
                ])),
                Felt::new(BigInteger256([
                    0x4488e34f67c8cd26,
                    0x6ff429f9162d9064,
                    0x6711b2edd0eeb7d4,
                    0x112546ac48abd018,
                ])),
                Felt::new(BigInteger256([
                    0x1cca826df45b541d,
                    0x2e564a342e78a567,
                    0x8e91ef7363cd24e9,
                    0x062033529df20cfc,
                ])),
                Felt::new(BigInteger256([
                    0xefbdbac257d1caa3,
                    0x98c12259a0a2455a,
                    0x59e123e67efbefb5,
                    0x05563cc78a2f9882,
                ])),
                Felt::new(BigInteger256([
                    0x4ad6987f0764032a,
                    0x7974beac3ade10b7,
                    0x641a3432437cd7ce,
                    0x00e83ec95978e820,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbf9579d892251f08,
                    0xacbd962979b965b7,
                    0xdde2a008d4914053,
                    0x03e68124e79e481b,
                ])),
                Felt::new(BigInteger256([
                    0x1f352ee9654ff4f4,
                    0x1042451303b33810,
                    0x0ccc035f0b26ff8e,
                    0x0abd6a1c6992670f,
                ])),
                Felt::new(BigInteger256([
                    0x6fe23cb6764fd992,
                    0x8da51f6e5bc9654e,
                    0x728c81e77000103f,
                    0x09cf953eb687cb55,
                ])),
                Felt::new(BigInteger256([
                    0xe7d59bd9c2d20956,
                    0xd959f29d555b3964,
                    0xe7230e5a2b6abbc9,
                    0x06324518301d16b3,
                ])),
                Felt::new(BigInteger256([
                    0xae678c9a713f16b1,
                    0x9f464d994f5dd3b3,
                    0x05fe57ea054eee22,
                    0x0fccdcc3993486ab,
                ])),
                Felt::new(BigInteger256([
                    0x75d296e56029986e,
                    0xd8a4dd7515d0609a,
                    0xe38753685bf2ec71,
                    0x0b123d77923b5ad8,
                ])),
                Felt::new(BigInteger256([
                    0xfc9138b0451e1d9d,
                    0xf9788852e36e8c1b,
                    0x91f43b44de6478db,
                    0x0f27b825b9464de6,
                ])),
                Felt::new(BigInteger256([
                    0xc68ca93e801d774c,
                    0xd27c70aee2f90c35,
                    0x1e0c522aa1bf5e46,
                    0x0959262ffe576a02,
                ])),
                Felt::new(BigInteger256([
                    0xa49db070e1cc4816,
                    0xf0aa9ea0c851c032,
                    0xf0b9f7646272b4a9,
                    0x0a5a68f651f00a7f,
                ])),
                Felt::new(BigInteger256([
                    0x5fdec8b1c1aa76eb,
                    0x0bac3fbe08071e02,
                    0xfe45ae0a5e7237de,
                    0x0dcfe7a4df40fcb2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0cfa15b05f2ae048,
                    0x68b7fda4d669e021,
                    0x1b4cfb209e044704,
                    0x02cada5edaa74b29,
                ])),
                Felt::new(BigInteger256([
                    0x4e0c18735d332a97,
                    0x8b3fac792fec150e,
                    0xe06d60fe981a9f7b,
                    0x0b5e20cc1750f946,
                ])),
                Felt::new(BigInteger256([
                    0x6700a8286d49d1fc,
                    0xf06340e6c16a5ec7,
                    0x6d81d26c223fe0eb,
                    0x03c105d8ca222915,
                ])),
                Felt::new(BigInteger256([
                    0x2e4ebec3d3275048,
                    0xd791d0df7af589dc,
                    0xcc9e8a3433acbf35,
                    0x0361dc9a40723b3d,
                ])),
                Felt::new(BigInteger256([
                    0x0b6aea9997620e86,
                    0x60bbdf34fed84a5a,
                    0xae244495c1c0dea9,
                    0x0655e3659615fdcd,
                ])),
                Felt::new(BigInteger256([
                    0x57b73aa8ad9d8e0f,
                    0x4666c72c1b715fb1,
                    0xb80c9769477e74ff,
                    0x0cc1928327b9528a,
                ])),
                Felt::new(BigInteger256([
                    0x7c4ea8ed454c08f8,
                    0x57a64729384a0ebf,
                    0xaca414ba2537fccf,
                    0x05d164671dddea7b,
                ])),
                Felt::new(BigInteger256([
                    0xff21def401ae23ff,
                    0x16ddfaf08c03ddee,
                    0x62af476582d3966d,
                    0x0ef394916e0b1eda,
                ])),
                Felt::new(BigInteger256([
                    0xb06020488c7b0a7b,
                    0x9fb11eaa6ac64611,
                    0x48a516ef1a7638d3,
                    0x092b193518cc1d46,
                ])),
                Felt::new(BigInteger256([
                    0xf47c203cff6e4c39,
                    0xaf702df6ee00e9fa,
                    0xe3baef46d3717ae3,
                    0x0c570f3defa02cf9,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x82f56cfd75d0cddb,
                    0xcd6682473cac336c,
                    0x025232db09238e9d,
                    0x04de298f317a8124,
                ])),
                Felt::new(BigInteger256([
                    0x96e60867ac9f6cf9,
                    0xb335b54e1c6c3fda,
                    0x4a3650721e33e17b,
                    0x0582c6970a7952ac,
                ])),
                Felt::new(BigInteger256([
                    0x6d60ac3138ac6684,
                    0x9c849de88bb03809,
                    0x4f9125537e2be637,
                    0x09b6ba1e4cf222bd,
                ])),
                Felt::new(BigInteger256([
                    0x10692a2c490248f2,
                    0xff2c3237916d4dbe,
                    0x655dfe35621ebeac,
                    0x0de45266f1b8bb09,
                ])),
                Felt::new(BigInteger256([
                    0x24f78943b7706410,
                    0x0b2a25c2c64d58e3,
                    0x45a5590f5cf02062,
                    0x11eb69289b1e061b,
                ])),
                Felt::new(BigInteger256([
                    0xc6c89f54b6f0c377,
                    0x43726426d4bdef3a,
                    0xa49b2db8debc6dfd,
                    0x075ef4b5d4a535c1,
                ])),
                Felt::new(BigInteger256([
                    0x93318bfbfb79899b,
                    0xbe441db95e9110ce,
                    0x8a9bf5a00452cd6c,
                    0x0b1f34d07b24272d,
                ])),
                Felt::new(BigInteger256([
                    0x3e8a9e9b180efdc5,
                    0x2944abe3f4b1a1c1,
                    0x84a4e2514488d5d8,
                    0x0f2b517869c88c58,
                ])),
                Felt::new(BigInteger256([
                    0xab8ee5cb99bd188a,
                    0xc50a0e5a719428a3,
                    0x170a3bf4c0f5bd45,
                    0x03136603a5bb40e8,
                ])),
                Felt::new(BigInteger256([
                    0x636a6ee1a17fc76d,
                    0xc1a35fd9eac850e6,
                    0xf88c1676858027dd,
                    0x05199a317f5c05fc,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe1ee9a04194325b0,
                    0x9afb1c06352a2755,
                    0x7148cc2e537dd9ef,
                    0x03315b0643c2ccfb,
                ])),
                Felt::new(BigInteger256([
                    0x47e1c4ca5c68f8aa,
                    0xd51b9ef227eb29b4,
                    0x2393750b2a77e626,
                    0x0df008d1fba00621,
                ])),
                Felt::new(BigInteger256([
                    0xd6668c6f644e4cf9,
                    0xd503b6bc3bea2bee,
                    0x6f3dbfcbab5f2c2a,
                    0x0085740032e1a3b4,
                ])),
                Felt::new(BigInteger256([
                    0xaf7b339aa206ba34,
                    0xd374f2bd07a7a020,
                    0x1ce0551f8fe97657,
                    0x074cad05b53a9927,
                ])),
                Felt::new(BigInteger256([
                    0xb1b67bc776486c69,
                    0xb44bac6d37aeb405,
                    0x6f5e74f641e492a3,
                    0x0bec39e552edb2b9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xef8320e28d90b41f,
                    0xf9cf72f83d717203,
                    0x14d03de799f57e2c,
                    0x11980bf5a1c0a325,
                ])),
                Felt::new(BigInteger256([
                    0xf4d14aec445dd7b1,
                    0x601ce785f9c563e7,
                    0xeeec3b935aa0e98d,
                    0x0844321ce6affe05,
                ])),
                Felt::new(BigInteger256([
                    0x92515a8ba292e42d,
                    0xbdcec66413ad2745,
                    0x02b9dc176eb891c1,
                    0x0832695d4828c5da,
                ])),
                Felt::new(BigInteger256([
                    0x12279c54c60908cb,
                    0x81c7e28c291e933c,
                    0x36f0ae2d3879c082,
                    0x11eb6100612e0bf0,
                ])),
                Felt::new(BigInteger256([
                    0xce75a43adcb96f83,
                    0x6ac161e1cb24e4ca,
                    0x6fe30ad1b38ff5e9,
                    0x0569c7b4a1140ccb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8ad9061ab2e4960d,
                    0xa4dc71faf3beec4d,
                    0x37acdb3698a1d2a3,
                    0x014648a503fcf616,
                ])),
                Felt::new(BigInteger256([
                    0x2d15681d485729f6,
                    0x4266499f62f42dca,
                    0x89ae992c2a37c742,
                    0x0de771e626493f3c,
                ])),
                Felt::new(BigInteger256([
                    0xaf1a281a7139ef67,
                    0x445e37829c6cdd65,
                    0x9c639f78c349b063,
                    0x0da0653c032bae11,
                ])),
                Felt::new(BigInteger256([
                    0xf869a69217a7582e,
                    0x69ec7a00bcb0913e,
                    0x0d03688807cb7ab9,
                    0x0dd246b2f118930d,
                ])),
                Felt::new(BigInteger256([
                    0x8085ab0a50361c87,
                    0x93b712f622645b52,
                    0x1aad4fda8aeeabac,
                    0x0e8de1ae5ad59dbd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3fc0f8ada316b1c9,
                    0x887ed6b5e8ec4208,
                    0xadb9d5cc64812f18,
                    0x02d234f3f8a86137,
                ])),
                Felt::new(BigInteger256([
                    0x32fda277b33ebfa4,
                    0x68e7bc5800cb29cf,
                    0xe449506ea7d54aaa,
                    0x0b361c7634b7503f,
                ])),
                Felt::new(BigInteger256([
                    0xd570fec8126eb3d3,
                    0x5c2c708da3582270,
                    0x9539f805d77ff6e3,
                    0x0a5bf3d44683efcb,
                ])),
                Felt::new(BigInteger256([
                    0xca07a8e41cb39124,
                    0x0bc6f3ea0ec0cf21,
                    0xd2ba81d4e589039a,
                    0x002742e40d5a8b07,
                ])),
                Felt::new(BigInteger256([
                    0x8fc1a264212fe48a,
                    0x95bf840ddb9d5186,
                    0x693dea17330087cc,
                    0x08a38c4b4b3150ca,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6dd71016271e9081,
                    0x8aef00a7703a9029,
                    0xe48a3a78952ca436,
                    0x06a6e60480ef9ea1,
                ])),
                Felt::new(BigInteger256([
                    0xaf0d3f698d855d6e,
                    0x3e4b50517bc204e5,
                    0x019f8f0069347465,
                    0x12755abd03f5181d,
                ])),
                Felt::new(BigInteger256([
                    0x696b3cd874cfa507,
                    0xe2dd0039b756a522,
                    0x0411564a3edbc20e,
                    0x109a2f838b92be96,
                ])),
                Felt::new(BigInteger256([
                    0x7a85c4215c40ddd5,
                    0x88181be4dc12bf4b,
                    0xc891539868eb1346,
                    0x0bde737e79e6654a,
                ])),
                Felt::new(BigInteger256([
                    0xe6281ff5d8663a62,
                    0xd0e81fc8627ba3b1,
                    0x0268b6622260c5c0,
                    0x0dd5338fb5037879,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcff4df25c09d680a,
                    0x378688adda6654c8,
                    0x7071daeec88301ed,
                    0x00a4e6468dd4ea5e,
                ])),
                Felt::new(BigInteger256([
                    0xabb465b81b42243b,
                    0x6a0dbc387a9ebfa2,
                    0x64e96b6c848391d4,
                    0x05fb6a5aea12408c,
                ])),
                Felt::new(BigInteger256([
                    0xeb16b67a97d17845,
                    0xc2d39e21da745851,
                    0xac31de392048806b,
                    0x0c2b3876d1a99055,
                ])),
                Felt::new(BigInteger256([
                    0x97bef1280a6b5a97,
                    0x5eca598ed41f5621,
                    0x1b5a18d60ee575dd,
                    0x042737db49f5b778,
                ])),
                Felt::new(BigInteger256([
                    0x637b4254eca0d946,
                    0xa5340421d8187d56,
                    0x284bf47e856d637f,
                    0x0c77f315dc734d82,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6918a2fe73190ffd,
                    0x5e230a81c7ca10c7,
                    0x5b6dd6866a4d89d1,
                    0x0feb8c381e6898c5,
                ])),
                Felt::new(BigInteger256([
                    0xb7a4d8cf00bc7f6f,
                    0xdd7f810d91533c32,
                    0x256a5a43e93abde6,
                    0x00a29151042f34ee,
                ])),
                Felt::new(BigInteger256([
                    0x23271e25ef11d8cb,
                    0x8ab78d39c37655a1,
                    0xded976d959e88a9c,
                    0x03b4220c1e8683a5,
                ])),
                Felt::new(BigInteger256([
                    0x5621c358fe3394fb,
                    0x1421e794a15dcede,
                    0xc3321c6e15ccbc35,
                    0x0a22738553319275,
                ])),
                Felt::new(BigInteger256([
                    0x0fee5eb9c107e515,
                    0x6cd79c319ba8b98b,
                    0x8a6e19cc611d5b18,
                    0x0d9ee351dd740092,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6c088a2ce5c0534e,
                    0x782da3cee80b6f5b,
                    0x76f8cfad0e317625,
                    0x093bc6d2c59e5f13,
                ])),
                Felt::new(BigInteger256([
                    0xfdeccf0fb22ce7f7,
                    0x288d580965f45074,
                    0xf7d3411dfb2c857a,
                    0x05bc51945cd063aa,
                ])),
                Felt::new(BigInteger256([
                    0x778d13eb57fa3a0a,
                    0x9c3565c55d213220,
                    0xb29291eb7c8b1d32,
                    0x010ccd355ac6fb8c,
                ])),
                Felt::new(BigInteger256([
                    0x9602080eea396b36,
                    0xf3b42a5484d032af,
                    0x19b7ab3cd41c14b4,
                    0x0d17e3fbd796f2da,
                ])),
                Felt::new(BigInteger256([
                    0x0c661e0c65dfd37f,
                    0xd33f551ee279ade4,
                    0x6c05fe4e78c15414,
                    0x038b92923027abd2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5e6d32aecdc7b755,
                    0xbc9ceb8ee9d1e0e8,
                    0xa789134f0e1173c1,
                    0x02fa722a0f2ff556,
                ])),
                Felt::new(BigInteger256([
                    0xb48d3d0e5e921a7f,
                    0x8348536492a191b8,
                    0x194cd43a8ab833fb,
                    0x02b74239bfdc73ce,
                ])),
                Felt::new(BigInteger256([
                    0xed5233e2ac7796d5,
                    0x704d87ce1abdde24,
                    0xf158890147448676,
                    0x03d5568d31cb5842,
                ])),
                Felt::new(BigInteger256([
                    0x289090c45e8bcdd6,
                    0x544af52ac3766cd1,
                    0x113d7812e7bf2885,
                    0x0d46be38d5128313,
                ])),
                Felt::new(BigInteger256([
                    0x09e5771b85455044,
                    0x243ea6de68713c77,
                    0x3cc1f51d61f0f1c1,
                    0x0ded614d7e679b6d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x57b400a5df92b1c2,
                    0x3835b8d66bb61772,
                    0x4345cfe0c774a308,
                    0x099410572557d6a7,
                ])),
                Felt::new(BigInteger256([
                    0x691a91c2e16ce9f6,
                    0x113f499689218f6d,
                    0xbfaef363a1ed9130,
                    0x093d4274a08c84c1,
                ])),
                Felt::new(BigInteger256([
                    0x703460b3cc9117d5,
                    0xeb5ed95eec4dc9a4,
                    0xad587537d6a77791,
                    0x0f6d03e4a826fdcc,
                ])),
                Felt::new(BigInteger256([
                    0x81a97ce16df86b92,
                    0x16e2c2900513d3f6,
                    0x261ced0ab142d46e,
                    0x061ac7666a43f29c,
                ])),
                Felt::new(BigInteger256([
                    0xa052e876e7755be0,
                    0x45f3b5a5d5b2607c,
                    0x360aabfa2b4eb71d,
                    0x03ed25874e7f80f5,
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
                    0x88747f6bae078127,
                    0x6abed4d6a46325d4,
                    0xc6fa8fe6b8cad146,
                    0x05b12bb9764ffc82,
                ])),
                Felt::new(BigInteger256([
                    0xc8dd467d5cb13ea8,
                    0xb194e52aae8250ac,
                    0x0002d0e53abe59ea,
                    0x0299ce87a89fc0ba,
                ])),
                Felt::new(BigInteger256([
                    0x01f906bad2a5bde3,
                    0x0f9146a1c9f0d5f1,
                    0xaf76573ef1565f9f,
                    0x1050916ba20a2acc,
                ])),
                Felt::new(BigInteger256([
                    0x44fa0a0f76333963,
                    0xd42d52b22aeee301,
                    0xb316d08b379c502b,
                    0x04496afa72fbf726,
                ])),
                Felt::new(BigInteger256([
                    0x52ff603a7ea5ae6e,
                    0x9b66d929000df00b,
                    0x4a2e50cc127e92da,
                    0x0be423dcb3b72438,
                ])),
                Felt::new(BigInteger256([
                    0x396958384d0cd73d,
                    0x06e58c38a331428a,
                    0x7084b4a263fbcd0f,
                    0x0fb147bb13d552d4,
                ])),
                Felt::new(BigInteger256([
                    0xdabe950c72d436aa,
                    0x8fdfd2a6e922ff7a,
                    0x1c4454e2db66dac0,
                    0x03736e80a07033c1,
                ])),
                Felt::new(BigInteger256([
                    0x45836366cb3adca1,
                    0x5c24b3b5f2b4d548,
                    0x5bb507bdf1529af3,
                    0x05e2f4447b9fba7a,
                ])),
                Felt::new(BigInteger256([
                    0x72014452b852ddee,
                    0x8af78fae98f7ea2d,
                    0xd0b3932d4bb0a638,
                    0x0bdfcc7b721c61bd,
                ])),
                Felt::new(BigInteger256([
                    0xa3255d9638b6b952,
                    0xec8ddd84a968e936,
                    0x11e7eda92266667b,
                    0x07ebfee3ba88725f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfa5c0c8ba88dc2f6,
                    0x3d075938caf8413c,
                    0xff7e8b75ccdacdcd,
                    0x00c8b85314c84546,
                ])),
                Felt::new(BigInteger256([
                    0xa5243353f61aa921,
                    0x411e998dfe3d1def,
                    0x93dd8af3e881c553,
                    0x0c785929c5ca4c40,
                ])),
                Felt::new(BigInteger256([
                    0x024e640f2c760eee,
                    0xba60d4e2f4a98f16,
                    0x45ba5ba9b35a02b9,
                    0x001dcaa2ec5751e0,
                ])),
                Felt::new(BigInteger256([
                    0x46997ff8906f5452,
                    0xfa5b19456c0b2dc1,
                    0x830062a910724a35,
                    0x0f4f6725ce23fe51,
                ])),
                Felt::new(BigInteger256([
                    0x5c6b429682eb3a01,
                    0x9309c873750f7d91,
                    0x35051e27d0f8dfac,
                    0x026c996dbddfe92f,
                ])),
                Felt::new(BigInteger256([
                    0x1ead41d030022e2d,
                    0x095a010f9a97b543,
                    0x08fb3cabfaf4103d,
                    0x0aa936eb74845422,
                ])),
                Felt::new(BigInteger256([
                    0x43a844d08c36c760,
                    0x8d6d38ee13a8f2e0,
                    0x099ba497d04962cc,
                    0x013d93a5186f7bb6,
                ])),
                Felt::new(BigInteger256([
                    0x0131c0195e3b7caa,
                    0x6f34cf412b34072f,
                    0x8b8f1d9053909500,
                    0x128264dc29b98793,
                ])),
                Felt::new(BigInteger256([
                    0x904049c34d9d8682,
                    0x3e301ccd546baf22,
                    0x84c4f7bcf26ec61f,
                    0x0585886c6042e448,
                ])),
                Felt::new(BigInteger256([
                    0xb8662aaaf2ef93b9,
                    0xcf7baaf6902cea25,
                    0xa6f795992dd2a8a3,
                    0x095ae6aa81e8c941,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xda5022688dd6c533,
                    0xe2ddfbcd543a7a7a,
                    0xaee6b84943e2eb81,
                    0x0c8f1858ee0fde65,
                ])),
                Felt::new(BigInteger256([
                    0x7470ace0479c26d7,
                    0xd9cf378e938c0e49,
                    0xb1865f75f076768c,
                    0x00b2871ff6372f4b,
                ])),
                Felt::new(BigInteger256([
                    0xd1186193c0fcdccc,
                    0xd42e59d76afd759b,
                    0x1b81b60352f32ec0,
                    0x02574bfbb8bc7b66,
                ])),
                Felt::new(BigInteger256([
                    0xa0580c423d9e8f06,
                    0x38e0eddc977e995d,
                    0x09b1d95a3c6826c8,
                    0x021faded93b8d53f,
                ])),
                Felt::new(BigInteger256([
                    0xef7857c27180bfec,
                    0x3a922c3316b3b707,
                    0xe4d46c91fe396604,
                    0x033be6136fcb9814,
                ])),
                Felt::new(BigInteger256([
                    0x4dc7fc311f7a6eb7,
                    0x9d5df6cd8c9fa45a,
                    0x32f4bc4fd4e3a306,
                    0x0c5548911e72d7ea,
                ])),
                Felt::new(BigInteger256([
                    0xdd91d0987816c5cc,
                    0x7a0d8fe3868e9566,
                    0x0f130fac8dc6de8c,
                    0x0ae9be4103332942,
                ])),
                Felt::new(BigInteger256([
                    0x287d492ea6cd51f4,
                    0x2605a3d1ad4dcb25,
                    0x1ef363aaca6cba94,
                    0x015c79b602b5f124,
                ])),
                Felt::new(BigInteger256([
                    0x4de2d7b84b4b92d3,
                    0x9d98fa1ff97362c8,
                    0x10d1a6bee5fa2a00,
                    0x02f8abc08d9c95bd,
                ])),
                Felt::new(BigInteger256([
                    0x0d9f859311fe13ca,
                    0x98480d5e01b32fcc,
                    0x89d4935c47bc875b,
                    0x060d9732fe1f613c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe481af50eeda8c65,
                    0xb21ee4363a070c8c,
                    0x41e1fa4117e6d973,
                    0x10e29a523bf04617,
                ])),
                Felt::new(BigInteger256([
                    0x798c0f0f8a11811e,
                    0x89eca7ccdecedce9,
                    0xa353906686acc1a7,
                    0x105765905a247df2,
                ])),
                Felt::new(BigInteger256([
                    0x4aa24663977a2720,
                    0x6ea58e2cd2c2920f,
                    0x76ceb1999d3760fa,
                    0x00a7c2974f36733e,
                ])),
                Felt::new(BigInteger256([
                    0xa245ecffeefd3e37,
                    0x4367d671a9495fe1,
                    0xaf53c73439af535a,
                    0x069c555a123a4157,
                ])),
                Felt::new(BigInteger256([
                    0xf8989f6bfbccf12b,
                    0x14dc63d06661a64d,
                    0xb35e58f6c061bd9c,
                    0x04d9e4ceb63d8120,
                ])),
                Felt::new(BigInteger256([
                    0x7ffe60110c4fd092,
                    0xe00cd9d431960b4c,
                    0x58fc01cc2dca7a0b,
                    0x0cf989559599b5a9,
                ])),
                Felt::new(BigInteger256([
                    0x890111b70ecfd3bd,
                    0x652bed70d826bf2b,
                    0x66e875508e567a8c,
                    0x014b70f9143403c8,
                ])),
                Felt::new(BigInteger256([
                    0x5dc3e68f5cd4eab9,
                    0x9f82562e651ed9a6,
                    0x626c26456dbcfe03,
                    0x0ff56799b7e81561,
                ])),
                Felt::new(BigInteger256([
                    0x9912878c6cc454e5,
                    0x0fea20b73b17e7d7,
                    0xe145e80780254099,
                    0x01185e1ac9aab1c8,
                ])),
                Felt::new(BigInteger256([
                    0xd8f659d812c328e3,
                    0xfa6421f84833bd18,
                    0xe172beb608ea4bf5,
                    0x10642fa0b318309d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x93f6194c78663b54,
                    0xf7a844536b554670,
                    0x82799bc696e2f365,
                    0x04995c105ac2bf17,
                ])),
                Felt::new(BigInteger256([
                    0xfa809908b23b8097,
                    0xa7dc353998f2983a,
                    0xaaaee1bae191ec8e,
                    0x069fdf8698b2aa2c,
                ])),
                Felt::new(BigInteger256([
                    0x0e7c5af26d4d6d19,
                    0x59be3a8d959c61e7,
                    0x282364b190f3d177,
                    0x04a49b3eb04d9136,
                ])),
                Felt::new(BigInteger256([
                    0x657828f651dcc16b,
                    0xe891ff8124c55e44,
                    0xa624c7896a305a7f,
                    0x02d908495e7ebe79,
                ])),
                Felt::new(BigInteger256([
                    0x9a3dfa4896907891,
                    0x818a6338d0e95dcf,
                    0xc119526030c90f6f,
                    0x07fc02a15ede22d7,
                ])),
                Felt::new(BigInteger256([
                    0x7fa7430364465298,
                    0xa2b069ea8b570d39,
                    0x17fc69d5834ed929,
                    0x1048379effd82a2d,
                ])),
                Felt::new(BigInteger256([
                    0x4bdca31bf68b14df,
                    0xe540c83b6594ff49,
                    0x9c7196834e9e1113,
                    0x05b2c40922ccf4b9,
                ])),
                Felt::new(BigInteger256([
                    0xe0e4439c1a8f207d,
                    0xf8937228884b3c4f,
                    0xc5c56c6424d6454e,
                    0x0f10c112ec6edf22,
                ])),
                Felt::new(BigInteger256([
                    0x9251474219753122,
                    0x7965d8999e720664,
                    0x70f7a2102bbc8006,
                    0x120b2b3f6fd12c0d,
                ])),
                Felt::new(BigInteger256([
                    0xaab027de45fc939e,
                    0xe03c6a21fb1137b0,
                    0xd5f46bee48810280,
                    0x01cc74cfd6a3857e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8d4e68268db7ee7e,
                    0x09e924c0670548ce,
                    0xaef243546f29f616,
                    0x109b8d4c016ccd11,
                ])),
                Felt::new(BigInteger256([
                    0x499365b4532dc5f4,
                    0x255a11a39614cb91,
                    0xdd2d003b40bf4b1a,
                    0x01401961850e548d,
                ])),
                Felt::new(BigInteger256([
                    0x44cd0c0ec78dd1c7,
                    0xbfbaf8dfa2163620,
                    0x7ddf19de1f285a1e,
                    0x0a56302001d2cec0,
                ])),
                Felt::new(BigInteger256([
                    0x445b72aaa0194405,
                    0x8a1bbdc25c931324,
                    0xda7fa9d9bedfd414,
                    0x0671bf456e9d9c1d,
                ])),
                Felt::new(BigInteger256([
                    0x245fbc79f67f0a0a,
                    0x1880bab18cad9347,
                    0x178156114ddb9ed5,
                    0x0e73dba5d7f92052,
                ])),
                Felt::new(BigInteger256([
                    0x1fdf9ea467b2ea3f,
                    0x9cc1fa023a586ca8,
                    0xc174ed6e8826bbd9,
                    0x0d3a10c257bcd92d,
                ])),
                Felt::new(BigInteger256([
                    0x3d1323cc8b293c07,
                    0x1702d79bf748e59a,
                    0xd41e1e0e2a5d13c0,
                    0x006ac27f14fcac22,
                ])),
                Felt::new(BigInteger256([
                    0x400b8a5601796d4c,
                    0xef308dbbcb975924,
                    0x7b9d8030b829d66f,
                    0x02fc47a5d72c2ca7,
                ])),
                Felt::new(BigInteger256([
                    0x9b030b67f263a609,
                    0x8805137c02848f9a,
                    0x11c6cf256e56dbfb,
                    0x02fd57369ac4349b,
                ])),
                Felt::new(BigInteger256([
                    0xb929ec079e400617,
                    0x46ec61b986deac24,
                    0x0c463cba1bb67a36,
                    0x019f993448082643,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x57571a9ff24991ef,
                0x733099e00855d11e,
                0x2fa47dfc9eeb453b,
                0x12345964e0401d5b,
            ]))],
            [Felt::new(BigInteger256([
                0x390e86ea1743e848,
                0xf7450053cf277535,
                0x8b2d27363ab19fe3,
                0x0161a00904558fbd,
            ]))],
            [Felt::new(BigInteger256([
                0xc1c367eed453241c,
                0x1c451b176234e40b,
                0x6352e4e3043660ab,
                0x012c180caada242b,
            ]))],
            [Felt::new(BigInteger256([
                0x97e76535a6a79aed,
                0x956f0494a76daeef,
                0x02813d0ea0284c0b,
                0x0e83af0f3242d7bf,
            ]))],
            [Felt::new(BigInteger256([
                0x402fd0a0ddbc61ea,
                0x9ddc282be94b2ac3,
                0x0029de882befd391,
                0x12402e249c1b1ffe,
            ]))],
            [Felt::new(BigInteger256([
                0x42e996a8cd765036,
                0x2f85c9fc4b8c1ad1,
                0xc2bf2290e54744e0,
                0x1013069dc927f8a6,
            ]))],
            [Felt::new(BigInteger256([
                0x692b4485b402bf04,
                0x925607ceccca8645,
                0x1ada9a7aadf330d4,
                0x054e336ab10aa52f,
            ]))],
            [Felt::new(BigInteger256([
                0x8ba7dd27bae2a8ff,
                0xc10e9096d2e88b01,
                0x13b928655ff13c3b,
                0x08368e9f1095f022,
            ]))],
            [Felt::new(BigInteger256([
                0xb8c72afcd635d00b,
                0xdc55dc92a683d6ea,
                0x5d52053e1f62d056,
                0x0c0f5e9f33fc386b,
            ]))],
            [Felt::new(BigInteger256([
                0x7a67e914c1d401b4,
                0xde3beaa36c6324b4,
                0xc2134117e86f157d,
                0x025ceeb9f0ac5753,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 10));
        }
    }
}
