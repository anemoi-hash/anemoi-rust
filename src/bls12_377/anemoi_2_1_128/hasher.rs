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
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger384([
                0x96aaaf5c1a40f552,
                0xab058bdbbdb60264,
                0x562fb8cb3212df3f,
                0x6d79ea140f8b8e1f,
                0xa8d19923bc6b354a,
                0x009233cd509cae3a,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x807ab92449e97c88,
                    0x2776d65c76205c70,
                    0xd3c8b20f5c02eb4e,
                    0x1a3162ec3e2db50f,
                    0xa2d7e796840a6908,
                    0x009e396ad5243ee2,
                ])),
                Felt::new(BigInteger384([
                    0xc1fd9cc3e3eaff02,
                    0x6452209bb4b1e2c7,
                    0xfa76175929d2a780,
                    0xb4a40b7a68247378,
                    0x1c85dee06384c97c,
                    0x0115e57b25f87df2,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x29b53aca69b55a9e,
                    0xbce803bde0eb62ae,
                    0x68d00c6eb5cf61f6,
                    0x4bc3e574a2dc28b9,
                    0x81fddaf7fd96ffaf,
                    0x00d1b78fdde0cae8,
                ])),
                Felt::new(BigInteger384([
                    0xad45030b449a0af6,
                    0x288b0953f306add8,
                    0x921b915bb01b0140,
                    0xd9c5005290d0ebdf,
                    0x81d248bc6dc32035,
                    0x00dd63f69ab37fc1,
                ])),
                Felt::new(BigInteger384([
                    0x3856c42f398e38e4,
                    0xee5c15bfa1550c0a,
                    0x0535e6c871217791,
                    0x66d8e4a478101348,
                    0x1c7dddb0457b3760,
                    0x007c617df9be96f2,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa3e56a8da2d667d5,
                    0x3eed79921638006b,
                    0x89517468d735a4c6,
                    0xb73ad5fd3569732a,
                    0x890d99b722332033,
                    0x0028af98111d890c,
                ])),
                Felt::new(BigInteger384([
                    0xf648f2467acda94b,
                    0x4ac1ed2a1d8c6f0f,
                    0x72dfd32fc6e2d989,
                    0x0e5a1617d24bc8cd,
                    0x8f14455984197f14,
                    0x016e6cbc84111dfc,
                ])),
                Felt::new(BigInteger384([
                    0x6d1524f08916ff17,
                    0x326f61164c052417,
                    0x167fc5830ca63b12,
                    0x1b54b970679e8751,
                    0xb2981ec416626559,
                    0x0090c38b57743674,
                ])),
                Felt::new(BigInteger384([
                    0xf53e2b12f8825d52,
                    0x4afa041c39e8c7d8,
                    0x61d933dde0b13308,
                    0xe3886b634ab96bf2,
                    0x3a2344209e7500b9,
                    0x00b472104a1dfa92,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x68d71f5481988917,
                    0xf635bfeaa76e521a,
                    0x3b8b7ae29d7eec82,
                    0x6ca28aebd0906d00,
                    0x52177fe1d6df5be3,
                    0x01aa0c2bbf206298,
                ])),
                Felt::new(BigInteger384([
                    0xb8512a9133e73902,
                    0x305e8e6e45d6627f,
                    0x4fe3bdd83c11402d,
                    0x0215ac531bc7888e,
                    0x64fc9e5b57cf819d,
                    0x00d86d140ca6a436,
                ])),
                Felt::new(BigInteger384([
                    0x2f3a178ab5f29678,
                    0xc0fca9c50a4ced22,
                    0x670d30bc04fee45d,
                    0xba0c18932b6d7a0a,
                    0x3d769351fbb47109,
                    0x00d8185a9ebea6df,
                ])),
                Felt::new(BigInteger384([
                    0x92c9e77fe2656934,
                    0x631f9c44ef832347,
                    0xbc1108aee380a113,
                    0x9f83d78845a234e6,
                    0x7e0321a32e7b1a24,
                    0x00aa51a50cbed191,
                ])),
                Felt::new(BigInteger384([
                    0x2935b5960a9a6fa7,
                    0xd6d160af7bf5344c,
                    0x0d96725143d2fa94,
                    0x43a146cb0aa5617f,
                    0x874b2cdd60534d99,
                    0x0024acbfad27c595,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x47de9665098c4b72,
                    0xc26d96a6f9df7f29,
                    0x3a8f7b77d8170612,
                    0x1bb9baaa0310d1af,
                    0x057f9cca8f3f1027,
                    0x006357466ec969b6,
                ])),
                Felt::new(BigInteger384([
                    0xa8c05e2e0974fd99,
                    0x1b08caa6779f9419,
                    0xd602aa8b5282bf53,
                    0x7493f6f050fc36e7,
                    0x187c7322a22d172a,
                    0x00f583c738de1999,
                ])),
                Felt::new(BigInteger384([
                    0x65747ff009dbf101,
                    0x9f085e820cd8b468,
                    0x8609897093cc840f,
                    0xcc88d5c364571bbf,
                    0xc612349e0e34972c,
                    0x0085e77100a57d4f,
                ])),
                Felt::new(BigInteger384([
                    0x2709e51e5bbfdfc1,
                    0xbdd84e655e071ca9,
                    0x104006f036121a88,
                    0x1c3dc3b1289e69c6,
                    0x72eb1e34acff8791,
                    0x015bc051a6009696,
                ])),
                Felt::new(BigInteger384([
                    0x582a9057c5fb9f02,
                    0x60e28d6a14c6ab86,
                    0xda232db5de42961b,
                    0x21ce16ebe767bf57,
                    0x3ab688b32ce01a36,
                    0x012dc5a6bb03d781,
                ])),
                Felt::new(BigInteger384([
                    0xa08f660cb56abe2a,
                    0x739c79edce0cf723,
                    0x6f12ff67ca2f4735,
                    0x944788ebf8579f22,
                    0xd38eac34b44fd0d8,
                    0x0097a0f16e5ba3df,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xa99a8aa05c76de7e,
                0x2b760e6bf8e0e989,
                0xbf316fffe9613ae8,
                0xa632bb4a945272bb,
                0x40260212411910e6,
                0x002eb6ccdd14cf8d,
            ]))],
            [Felt::new(BigInteger384([
                0xe7b3e9ee65da45c8,
                0xb7b12810e3a7d55a,
                0xa6736bb5b167fed4,
                0xad55467ed2dd743c,
                0x6b1c3cb4430afda9,
                0x001da93ebc81fd68,
            ]))],
            [Felt::new(BigInteger384([
                0xb0677c22a7ec1794,
                0x1cd033757163e8c1,
                0xdfee1074f61fccb5,
                0x4d4fadc5324cb85d,
                0x5a39f717c6cf15fd,
                0x01980c86060c8a0e,
            ]))],
            [Felt::new(BigInteger384([
                0x8544b031900fef97,
                0xa495500106f686af,
                0x8d48bd22cc29b861,
                0xe9d1e3e4cec8c074,
                0x58b1e2c58f674249,
                0x00e1679eeac7ba8e,
            ]))],
            [Felt::new(BigInteger384([
                0xc9c5b787b113c800,
                0xbfeb82d96394dfb4,
                0x17e9af0649647fd4,
                0x9b1c80e6b6ada8fe,
                0x9f9d0c0ffdd47a96,
                0x01714f61297f4808,
            ]))],
            [Felt::new(BigInteger384([
                0x7fbb8b1993d0c091,
                0x3cb6f270890fd55c,
                0xd910b2b3cb6e0efb,
                0x912e4c24a8321a22,
                0xa3d1ab53c058bc25,
                0x01845b9cc1411130,
            ]))],
            [Felt::new(BigInteger384([
                0xb9cb87880df2e404,
                0x327800984aa5e65b,
                0x95d90c87dd26260f,
                0x96ddc7ff0a429fff,
                0x3ee1a6aa5dfa1799,
                0x014151c0e46ee7da,
            ]))],
            [Felt::new(BigInteger384([
                0xa4e90ab92972e439,
                0x28caae2c4c67516e,
                0x717430be76764a54,
                0x7b17903796930e6d,
                0x3418e028a3206994,
                0x01420f63e56321be,
            ]))],
            [Felt::new(BigInteger384([
                0x96c76f18738c1b06,
                0xdfdaf2c69a018391,
                0x120a44cfda4ad81a,
                0x3a166064de5647dc,
                0xe213f3169ae8aab5,
                0x00ae28209ec40963,
            ]))],
            [Felt::new(BigInteger384([
                0x075924fd8ea84ff7,
                0xb35e879368849bb7,
                0x8981ba4529aefe0c,
                0x8d02688108ff5526,
                0x36317d67b487f455,
                0x002c710d4fe5cee0,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![
                Felt::new(BigInteger384([
                    0x101a57b166c0ddf9,
                    0x72c9b24555de70c3,
                    0x5ace909a8223e58a,
                    0x5745d3a427871ff6,
                    0x41da1acba4e32b60,
                    0x00f03ca2a104afed,
                ])),
                Felt::new(BigInteger384([
                    0x4daf08469948087a,
                    0x988355d16ce2217e,
                    0x62f79533f2d9ac96,
                    0xeabfcfc02b00d615,
                    0x7835b4f5559849e8,
                    0x01a0937c6f0f4f99,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x195d2b449a87060a,
                    0x67d51d1f1216549b,
                    0x17b12bf4a93a7707,
                    0x68b33f41682e1c05,
                    0x4461ad9ceff7f4ef,
                    0x01a2cc3054c92fd8,
                ])),
                Felt::new(BigInteger384([
                    0x57dfa7660bdceb76,
                    0xcaaf480dc281d464,
                    0xff0b67c7e953f4a7,
                    0xfc1aa7c312d16224,
                    0x3887c8ea97aab8b2,
                    0x00cbdf4627fb8aa0,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x6dc48c998c83babd,
                    0xd4ed4e795e67c778,
                    0xf09cee85e8ca89f7,
                    0xe683294c21c40e5a,
                    0x639cd23c8253a06b,
                    0x00bc6c910bf571a7,
                ])),
                Felt::new(BigInteger384([
                    0x8ed31fd35dcb46a6,
                    0xc9053f3b5c91bc78,
                    0x6463eb50d3791d42,
                    0x5b238d403741ca4b,
                    0xa51087f8c1c834be,
                    0x003645b4c677315d,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xda17e3d96d147a37,
                    0x97b21c9347b5864b,
                    0xa3d31f77fe59e08c,
                    0x9edcd453dff00cf4,
                    0xaa0b72019301f617,
                    0x00658aec29a1a41e,
                ])),
                Felt::new(BigInteger384([
                    0xbb6b12d4493492a2,
                    0xdafbeec3e44b1786,
                    0x4f2ff9de8eb24931,
                    0x934210dadbdd85aa,
                    0x58524e61bbccd8e3,
                    0x009ae7879f9eab35,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x7e81ac1fbd155cef,
                    0x3013d5257e0d44a5,
                    0x3e8c88c32de71d15,
                    0x6d627641c7264f57,
                    0x3841eff365b74d51,
                    0x00192654384b68c2,
                ])),
                Felt::new(BigInteger384([
                    0xa909b8f1c0e125df,
                    0xc7383cadd6894790,
                    0xfbd441843215080e,
                    0x29f153c5fa261d41,
                    0xe0aec10099e7e702,
                    0x009d1ec59aae5f75,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc9400e24abd131a4,
                    0x3caecc17eb87bd57,
                    0x8ca704829ea73a39,
                    0xb3fb3d9fac8dbee1,
                    0xb247c09e4df603e2,
                    0x0012ede4a25f246c,
                ])),
                Felt::new(BigInteger384([
                    0x261336dd0b1c4718,
                    0x6ce3b4c8456ece83,
                    0x11aefffcbbe0b159,
                    0xcbec018880f131d4,
                    0xbbfac72567f58ace,
                    0x0111483f2e97d6b4,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xe50c406a9690ccff,
                0x063bc189dcf29232,
                0x73c981588effc435,
                0x4e060aa721d6efa3,
                0x766e2a4b4a41f671,
                0x001a48b29a110f3b,
            ]))],
            [Felt::new(BigInteger384([
                0x2810994b36b7001e,
                0xe9fd644e40e57d57,
                0x111e7a90f729d4a0,
                0x75d6015d907e4f0b,
                0xdf4aa50016c25e81,
                0x012acc367dd4ecf2,
            ]))],
            [Felt::new(BigInteger384([
                0x5d662fbc8ebd493c,
                0x527bbaf9a6acf40c,
                0x047ff1e4ff139ece,
                0xcadd1ba6be56227f,
                0x038a137ed94b12d5,
                0x008254149f7897c8,
            ]))],
            [Felt::new(BigInteger384([
                0x200d1694b28c4e75,
                0x4cd74901db0e5252,
                0xd5764d2164f99276,
                0xed03b1b505ea05fb,
                0x802380f0f001acbe,
                0x013a7917a78b4b87,
            ]))],
            [Felt::new(BigInteger384([
                0xe321c419498f7a9d,
                0x66a99f51a2412d6a,
                0x098a742a17a32429,
                0xe470dd5567e7d093,
                0x7c8ba724fe8e5876,
                0x01a15e037f187b74,
            ]))],
            [Felt::new(BigInteger384([
                0x6146922eea937d72,
                0x4f6b6f992c311511,
                0xa62b531b1949dfcb,
                0xcfe96f3284e715f9,
                0x88a770aa620fa9da,
                0x0165636114f8e6d1,
            ]))],
            [Felt::new(BigInteger384([
                0xf69cedbbee81a5df,
                0x9a8f2a638eb83fb3,
                0x0a9d0fd412ab61b9,
                0x4b9bdb0b5cc43935,
                0xa02bd8d5ff6b76a5,
                0x0168856a87236e80,
            ]))],
            [Felt::new(BigInteger384([
                0x9686c90406c4f768,
                0x302e2b4639d2d79b,
                0xcc5609fe818ba95f,
                0x253294d6cfff1c65,
                0x8579a776b660f471,
                0x001306eaacc0f7c3,
            ]))],
            [Felt::new(BigInteger384([
                0xd762c8b8e765487c,
                0x86e226f929e06ea7,
                0x966952bba53f0972,
                0x075480ffe2318a22,
                0x135c1b95301260dd,
                0x0089938045ba2c57,
            ]))],
            [Felt::new(BigInteger384([
                0xaeff298d5fd8e25a,
                0xcf42a04c4cbebed3,
                0x3cd5e66bd1e7eb87,
                0x132ee9c61c23f35f,
                0x298b243523dc1aa8,
                0x00042abecaa6aec7,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }
    }
}
