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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
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
            vec![Felt::new(BigInteger256([
                0x51588956acff187a,
                0xde6221e3a106d49f,
                0x487d442f4be11175,
                0x29f468b45daadb1f,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x27a5a6f0beaec9a3,
                    0xe654fc5317bc5d66,
                    0x97feca42116f2d96,
                    0x0b7c3c45879110d1,
                ])),
                Felt::new(BigInteger256([
                    0x54623e808abfdd5d,
                    0xbb3d347caf4124a9,
                    0x820c15c6fc747f9a,
                    0x0bc62b0669ee023f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x706dc364f64f7441,
                    0x4bd81de69c2ffca2,
                    0xdb195fd41fef2fae,
                    0x1124617afe96140c,
                ])),
                Felt::new(BigInteger256([
                    0xd6b742775160714a,
                    0x360042d085f6518f,
                    0xf0d6c6bc3658a7b6,
                    0x2dbeea95f6e4030a,
                ])),
                Felt::new(BigInteger256([
                    0xebc115acb2a38d7a,
                    0x9f4efea24d649f3f,
                    0xeedc9d2a105ca20a,
                    0x32efa64a1e015d2b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb4c02896848fc6da,
                    0x9f9056f959238c19,
                    0xa01f32753c75aabb,
                    0x1a129bacd2df30c7,
                ])),
                Felt::new(BigInteger256([
                    0xd8dff7b4004ef416,
                    0xdbb2efb4a434c5ee,
                    0xf56cf20ad938f50d,
                    0x12a060fd0c728438,
                ])),
                Felt::new(BigInteger256([
                    0xb3abd14f618859dd,
                    0x8fd6989cbf874c9f,
                    0x7e1bc0595b0f1b2a,
                    0x070d465e3eddb451,
                ])),
                Felt::new(BigInteger256([
                    0xbf17b9413f13a9e3,
                    0xa3fab94c1a13388d,
                    0x2aec97edf67e0edd,
                    0x3697b35ae2281897,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5416e49d0a551070,
                    0x57c78b40542ee29e,
                    0x37b3f9c0c8aa63b5,
                    0x05613f250058a91d,
                ])),
                Felt::new(BigInteger256([
                    0x36f708f3b33d5923,
                    0x9af7727fc0235f41,
                    0xb85549c5e9fb53d0,
                    0x216b1adeb502d890,
                ])),
                Felt::new(BigInteger256([
                    0x1ee499ebc4f5b7a5,
                    0x61176e596a585721,
                    0x5763098e6f212c30,
                    0x2798cb7fa2a19805,
                ])),
                Felt::new(BigInteger256([
                    0x96b8f231654cc60c,
                    0xf19daff8295994bf,
                    0xaa624d2976a0ac6a,
                    0x11db342632114338,
                ])),
                Felt::new(BigInteger256([
                    0xa9c7fe28162b8bee,
                    0x9d533efc7d2a9657,
                    0xc525bd20beddc07d,
                    0x339bea8ba292d534,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xba9ccdb0708b3ab0,
                    0xe2fa96efa4a0f061,
                    0xf884b2e09b770526,
                    0x02238ad0de72bb43,
                ])),
                Felt::new(BigInteger256([
                    0x02d330df596a0aba,
                    0x3f0ea1c9ea7ace26,
                    0xdb976c1b9b574bf5,
                    0x0bec1d103a3e599b,
                ])),
                Felt::new(BigInteger256([
                    0xc6124da318db3171,
                    0x996211ccfc76e96c,
                    0xab6e8ce1e636e6de,
                    0x343b1785b0180ec7,
                ])),
                Felt::new(BigInteger256([
                    0xc0a3766a8514b827,
                    0x8554d32c8a054ab0,
                    0x2d9db059edfcc5ea,
                    0x31e1fbe26cf82499,
                ])),
                Felt::new(BigInteger256([
                    0x557e193d60579a35,
                    0x9c3383ff73ffa655,
                    0xf9ba823ffc29adef,
                    0x0508bdd5009e2b1c,
                ])),
                Felt::new(BigInteger256([
                    0x5486bebde10a07a0,
                    0x4a27bacd6c8d6126,
                    0x821cdb6894403e17,
                    0x194051a297758d8e,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xe121d5c718a0559f,
                0x91185b11a80c626c,
                0xdf981b70449c0826,
                0x264b2bda05d11256,
            ]))],
            [Felt::new(BigInteger256([
                0xc2e9b760fda924bb,
                0x65dbea044959d455,
                0xcd5b5edd10b08cfe,
                0x01e913e513d3cf43,
            ]))],
            [Felt::new(BigInteger256([
                0xb7e157ccfdeeea77,
                0x7dfd53dd5ed5cab8,
                0x878bc23b5b0c7ce0,
                0x0f64e29108d7206c,
            ]))],
            [Felt::new(BigInteger256([
                0x1114557bce0442aa,
                0x7b5e40e001057ad2,
                0x708b997206277c41,
                0x14a167858fcaf5e1,
            ]))],
            [Felt::new(BigInteger256([
                0x84ed80edef37091a,
                0xee266ae80ba9cb10,
                0xbec0437f9b7c7ad7,
                0x3298c61a432ee7f2,
            ]))],
            [Felt::new(BigInteger256([
                0x04be077c764c0f43,
                0x12654e745c93a251,
                0x30ae62d883425629,
                0x3289654d666c2122,
            ]))],
            [Felt::new(BigInteger256([
                0x0127d4470df3e33f,
                0xf83d73eb4d231d04,
                0x77180fe074c9393e,
                0x296020fadd1e74c5,
            ]))],
            [Felt::new(BigInteger256([
                0xdeeebcf3f9c83338,
                0x946a4915688c1659,
                0xccec9813a6b8f1a7,
                0x18542c19615be0c4,
            ]))],
            [Felt::new(BigInteger256([
                0x2c6dfbae47c383e8,
                0x4384441c496fc2d4,
                0x30e0cb4c5a73e276,
                0x14b2d123e68a0d6c,
            ]))],
            [Felt::new(BigInteger256([
                0x7f2d5c19c96286b2,
                0xe45dd7a1d5b19fce,
                0xcb543edf24d34c36,
                0x086a45c8137eece5,
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
            [Felt::new(BigInteger256([
                0xe121d5c718a0559f,
                0x91185b11a80c626c,
                0xdf981b70449c0826,
                0x264b2bda05d11256,
            ]))],
            [Felt::new(BigInteger256([
                0xc2e9b760fda924bb,
                0x65dbea044959d455,
                0xcd5b5edd10b08cfe,
                0x01e913e513d3cf43,
            ]))],
            [Felt::new(BigInteger256([
                0xb7e157ccfdeeea77,
                0x7dfd53dd5ed5cab8,
                0x878bc23b5b0c7ce0,
                0x0f64e29108d7206c,
            ]))],
            [Felt::new(BigInteger256([
                0x1114557bce0442aa,
                0x7b5e40e001057ad2,
                0x708b997206277c41,
                0x14a167858fcaf5e1,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 186];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);
            bytes[124..155].copy_from_slice(&to_bytes!(input[4]).unwrap()[0..31]);
            bytes[155..186].copy_from_slice(&to_bytes!(input[5]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
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
            vec![
                Felt::new(BigInteger256([
                    0x47f3c46cb09932ad,
                    0x1e46ed5f52c9551b,
                    0x04c62acb05fd739c,
                    0x155481423b67c474,
                ])),
                Felt::new(BigInteger256([
                    0x40f3c32104802ec0,
                    0xd5978b7a6f876a87,
                    0xefceaccba1d77235,
                    0x2ffa991724f15d03,
                ])),
                Felt::new(BigInteger256([
                    0xa7f614aa549a1143,
                    0xebc7126dc647987e,
                    0x6695bc3238a68ddc,
                    0x350f32d63f425fda,
                ])),
                Felt::new(BigInteger256([
                    0x3f4c6b37a95a4a40,
                    0x38650a4d734fbe85,
                    0x8c527596b9bc56aa,
                    0x01d87aa839effd32,
                ])),
                Felt::new(BigInteger256([
                    0xf8ab4f1e63e6f153,
                    0x0f28da7c5da50893,
                    0x9bd79f01cd96d4d5,
                    0x0bd1cf4f54948d74,
                ])),
                Felt::new(BigInteger256([
                    0xeaab7dcce20311f6,
                    0x37d407aab3a1db99,
                    0xb58c1ced2340f4a4,
                    0x089c117cbb29301a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4613eadb7866b0a7,
                    0xee8e4843c5fb598b,
                    0x5925ac63de5097dc,
                    0x3867e3bf49f149ee,
                ])),
                Felt::new(BigInteger256([
                    0x28bf1dc288fccf4c,
                    0x5e0e308f17ed0962,
                    0x980a1e80cd4b7439,
                    0x3664b9b2ecd9ef0b,
                ])),
                Felt::new(BigInteger256([
                    0xbc8d12754ba1d4d7,
                    0x7242655da5e2cdd3,
                    0xf2115bd0cadf9f4b,
                    0x27c1e32536bbed9a,
                ])),
                Felt::new(BigInteger256([
                    0x373886a915dc3852,
                    0x0aece3af80cd6214,
                    0x3d1c236dffd2880c,
                    0x0769726f7757837d,
                ])),
                Felt::new(BigInteger256([
                    0x07679ea76bf52b50,
                    0x0c09bc0043f44dc0,
                    0xefe61fa74e63c954,
                    0x2bce15acafa06a0a,
                ])),
                Felt::new(BigInteger256([
                    0x32aded3295650be5,
                    0xe8ab73e70e25adba,
                    0x98aa26ab90d4e554,
                    0x2b18d4dcf854d448,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7f1f3f8d3a557f0e,
                    0x633b1471aa55f4ef,
                    0xf97ca47b03d4b666,
                    0x3af6e85c319a7389,
                ])),
                Felt::new(BigInteger256([
                    0xe2aaa62cf9ca74f5,
                    0xcd3990e6619d15dd,
                    0xe6d641368c41757c,
                    0x0bd6d1662c8962ca,
                ])),
                Felt::new(BigInteger256([
                    0xe954f1b034a717ae,
                    0x9e2cc87bd8f61beb,
                    0xd33d2b0d5debf262,
                    0x253a49937acbf87c,
                ])),
                Felt::new(BigInteger256([
                    0x4b10490298ff86e1,
                    0x1fa05ee172c25358,
                    0x143e2789e3751da5,
                    0x384310ea035e9b36,
                ])),
                Felt::new(BigInteger256([
                    0x6ca7ce672483b9ca,
                    0xb007cbdc152f4ace,
                    0xe9ddcbae6a9d39e2,
                    0x222a277958c2f90e,
                ])),
                Felt::new(BigInteger256([
                    0xef12384d378714d1,
                    0x5e9ca50ce1d08b04,
                    0xfb11ec5ae80505e9,
                    0x2fd49bd78844c6d3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x167976a3e6a86d89,
                    0xcde7b42aca7b6e68,
                    0x2787dfd26105e09c,
                    0x0715346e3e287483,
                ])),
                Felt::new(BigInteger256([
                    0x72ab2db6f7ec8550,
                    0x574c164ce07c0def,
                    0x4b9f9ed7c9b17d46,
                    0x16604c42534bc0bd,
                ])),
                Felt::new(BigInteger256([
                    0xa67025be75b671b6,
                    0x4389a31029cd9328,
                    0xb6a4207873f23587,
                    0x3832b686d9f0e6dd,
                ])),
                Felt::new(BigInteger256([
                    0x0b36a074b82f43bc,
                    0x52d4b0b984a56022,
                    0x76de619b0dedbd1c,
                    0x1a3ad21215ff9c60,
                ])),
                Felt::new(BigInteger256([
                    0x60a0dac5cd214b25,
                    0xa9a1741c15bf7c0f,
                    0xb093078846fefd9b,
                    0x0f8e71c136d4c8bc,
                ])),
                Felt::new(BigInteger256([
                    0x9817832ce9048c41,
                    0x673280508e16f024,
                    0xd3b3d0d888c1ad31,
                    0x1c88b4f8bfc65943,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x504e12dd855405db,
                    0x8fa598b2886550bd,
                    0x22f6c2a97938eb79,
                    0x11dcd2d5ae61eac1,
                ])),
                Felt::new(BigInteger256([
                    0x13de976b8bc1dccb,
                    0xae24589615726be1,
                    0x6d89ecae9b50d103,
                    0x3cf00b32e876a028,
                ])),
                Felt::new(BigInteger256([
                    0x3507b8a7a68014b6,
                    0x6cf8f34071ad61a6,
                    0x8e08745bc91b4aa8,
                    0x1812ba3cd56fe201,
                ])),
                Felt::new(BigInteger256([
                    0x853104650033bf49,
                    0xf4653038d4f6a630,
                    0x3df92b39846bba21,
                    0x045e0f71ee2d6fc7,
                ])),
                Felt::new(BigInteger256([
                    0x0497e1ea87d49f36,
                    0xac138389c4b5c259,
                    0x9b0386596c939df7,
                    0x12e8085065bc7063,
                ])),
                Felt::new(BigInteger256([
                    0xd1cee66b4c2e7447,
                    0xa8e552e6147539ad,
                    0xac135beabb93e9cb,
                    0x340ab1022a7647fa,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4bfb618a451322bb,
                    0x28cd542aad27e8c6,
                    0x07950d57ed7c9ecb,
                    0x06744c38b4b7ece1,
                ])),
                Felt::new(BigInteger256([
                    0x086bbdf4cf204310,
                    0xe322c70f7be3eba7,
                    0x67c84097661b7b45,
                    0x241837648e5375b0,
                ])),
                Felt::new(BigInteger256([
                    0xf61c2f463254fe7e,
                    0xc1abc076b08eb3fc,
                    0x397a7f24e6806413,
                    0x341c4d25d3f04bde,
                ])),
                Felt::new(BigInteger256([
                    0x38e535b4061826cc,
                    0x8f463c25712cf22e,
                    0x9c4c78d6fcd4d67d,
                    0x0e9ab8ac02acc046,
                ])),
                Felt::new(BigInteger256([
                    0x5803cf5d554e6f21,
                    0x2e8bd1f076ab6d77,
                    0x00c0d8031ddc806a,
                    0x3664701a5ce0129f,
                ])),
                Felt::new(BigInteger256([
                    0x1c8df9c52072abee,
                    0xb8e68bc3d339549a,
                    0x52ec6c9690671506,
                    0x0fb79e97afd4026c,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x99602edc0ad2e282,
                    0xdd342b953bc452a0,
                    0x3c0e7efce1a88600,
                    0x317047b95195155b,
                ])),
                Felt::new(BigInteger256([
                    0x1313b8751ac4b7a2,
                    0x52ae791c8b125341,
                    0x51b1d551b9c2dfcf,
                    0x1f0eea8fa3f6603d,
                ])),
                Felt::new(BigInteger256([
                    0xe40d83c82bf4037f,
                    0x3c1a04488150dde7,
                    0x2110e15d10f3ad35,
                    0x1f3342bdd9b6dab8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbde4ac23f6e54c09,
                    0xf095adf366717db7,
                    0x4205df68978e4b6a,
                    0x06246fcaf12ffb53,
                ])),
                Felt::new(BigInteger256([
                    0xb715c53107c355ca,
                    0xb5870f8d873c7710,
                    0x25a831e957064b37,
                    0x0b3b6df488d97d15,
                ])),
                Felt::new(BigInteger256([
                    0x5b963374f846255d,
                    0x7c254829caae9acd,
                    0x9e4317edb03fb9a5,
                    0x34d122c2543cf816,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc12340b93b3cca5d,
                    0xe33af3fad12e2fbb,
                    0x4ae8575064ddbaea,
                    0x211359ce7c7c95bf,
                ])),
                Felt::new(BigInteger256([
                    0x51676cfd0689254b,
                    0x1a08112801b3ea36,
                    0xef12f9190bd7d419,
                    0x37023846922f7093,
                ])),
                Felt::new(BigInteger256([
                    0x38c22d826efe4d9f,
                    0x2db3abf7d9db856c,
                    0xe95917c31d904ec3,
                    0x10bd402f304d9418,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1600f475f4964b42,
                    0xa32b9abb8d46dbed,
                    0x54efea154ef2e15b,
                    0x309168336b2cb2c7,
                ])),
                Felt::new(BigInteger256([
                    0xf199e56c79609d33,
                    0x2cd1bc1c8627d26b,
                    0xbadad08746bd3a95,
                    0x2ce6d4de8f2011d4,
                ])),
                Felt::new(BigInteger256([
                    0x84f7891d6835956b,
                    0xfc2fd7911ddc1905,
                    0x05176707ab4a3c85,
                    0x2a338e2939eba088,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x77ae18147b5795cc,
                    0xfc400690d5317f19,
                    0x7eca94e87bc2b523,
                    0x35d3f0a8f911bd40,
                ])),
                Felt::new(BigInteger256([
                    0xba01c2f09c05f4ec,
                    0xe7312d4ad9cc6e48,
                    0x27f532ccf6d93262,
                    0x1c5f1ed6a3279864,
                ])),
                Felt::new(BigInteger256([
                    0x55685cf8208765f3,
                    0x02ff4205ed262f88,
                    0x329aee73400b1b72,
                    0x22c0c32a0efbfc72,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe1b4b63e93a99be1,
                    0x70090b3897482e31,
                    0x9ffeb833342859bf,
                    0x32e9a76f9b96fbcc,
                ])),
                Felt::new(BigInteger256([
                    0x53b53da37ad8c16f,
                    0x2773fd0472260c26,
                    0x8d3c3c8545c94def,
                    0x2d41de660903df21,
                ])),
                Felt::new(BigInteger256([
                    0x82eba4cd0cd59265,
                    0x15fa0eea935c62f4,
                    0x4d38cfa918cb142c,
                    0x08077cf5a0009d3f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7a53417a4a7dee6b,
                    0x00ea5422155928dd,
                    0x41f357d9774f3f0f,
                    0x30fcaa86e26d4b3c,
                ])),
                Felt::new(BigInteger256([
                    0xb3d96b8e61480639,
                    0xcf057300e30fb42a,
                    0x7b4a3615b953458a,
                    0x0f2ee135190e9860,
                ])),
                Felt::new(BigInteger256([
                    0x01835842fc575a7c,
                    0xcb226eba4319850d,
                    0x7832e7574d47c290,
                    0x03b9a8f6050e1ab6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5cfd031008be3fed,
                    0x2ae58aef7e51a5d5,
                    0x456c4d98ec42fe01,
                    0x0a4edf778d9c0a2c,
                ])),
                Felt::new(BigInteger256([
                    0x0742f554f94f4c4e,
                    0x1364e3e1cca764fb,
                    0x7d7372607c8303b0,
                    0x1be54a538b5be896,
                ])),
                Felt::new(BigInteger256([
                    0x0d597c9cb7f003e1,
                    0xeafc5db36109974a,
                    0x285bf64426376a74,
                    0x27c9e9d436d296c5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6be6ee4001c8c84e,
                    0x6c7b01908ca07502,
                    0x67dab2a552fc96a8,
                    0x223341d3c051c8b0,
                ])),
                Felt::new(BigInteger256([
                    0xb0026545c2461712,
                    0xa0b233059af2bbda,
                    0x2d06f54cc7608420,
                    0x3f3e1409c7537476,
                ])),
                Felt::new(BigInteger256([
                    0x89979b4888f73714,
                    0x88856dea32c7c6ac,
                    0x9703eb2ba2f89663,
                    0x1362b9f9b826f7d4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfdea27dc819b36a0,
                    0xa30b222b968a9da5,
                    0xe33710521b33a66d,
                    0x0080315e428122aa,
                ])),
                Felt::new(BigInteger256([
                    0x82a44edf2c5d9b8a,
                    0x33e6398de17efc8f,
                    0xdcef6a6a311be075,
                    0x391c50c2981adf63,
                ])),
                Felt::new(BigInteger256([
                    0xb2aa6e403441e0c8,
                    0x1d2e950ab5bd22d1,
                    0x3fd4115e045cb447,
                    0x058cc3a19b103506,
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
            vec![
                Felt::new(BigInteger256([
                    0xecf55d0b5e646c59,
                    0x59986012e2e0ed5b,
                    0xcafbc08104fe5af3,
                    0x08eae0630c3d00c6,
                ])),
                Felt::new(BigInteger256([
                    0x6524d04c279bcc00,
                    0xdfa2458773691533,
                    0x925036f448f602c2,
                    0x0f5a91899b190247,
                ])),
                Felt::new(BigInteger256([
                    0x585464610b21390d,
                    0x9ec53591b0c5e93d,
                    0xaf31a8b768d80b5c,
                    0x2e0ee9fa713dd097,
                ])),
                Felt::new(BigInteger256([
                    0xa4806304903c4b95,
                    0x760ee7b6cb9d6754,
                    0xef0d22522b6ec3f9,
                    0x1a8ae7368223dc01,
                ])),
                Felt::new(BigInteger256([
                    0xa839b4574db43fc1,
                    0xa7a200f06986edef,
                    0x8da363543ecf674e,
                    0x2d4987ed918f5204,
                ])),
                Felt::new(BigInteger256([
                    0x8c1b68d33d5fe415,
                    0x74ad6b1aed15b29a,
                    0x66d73dc55ada76a4,
                    0x3e3022fe7db2ada0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x606c9b2814bded12,
                    0xae47560a32aed04b,
                    0x56d8af0f6d69979a,
                    0x26c4ef0f6fb80c79,
                ])),
                Felt::new(BigInteger256([
                    0x85daddf62b38acf4,
                    0x67b89de4a1d832cc,
                    0x6042c251f67bd92b,
                    0x2c3cc1edf2303384,
                ])),
                Felt::new(BigInteger256([
                    0x0df1b225997cda5b,
                    0x45a611d8834049cf,
                    0xc5165bac268daca9,
                    0x1e877c511d1acd1c,
                ])),
                Felt::new(BigInteger256([
                    0xded01bf007bafb9e,
                    0xaf1e2d07013b9d79,
                    0xa49e314e932849bc,
                    0x0b289a65107ede94,
                ])),
                Felt::new(BigInteger256([
                    0x54e55285d0691ec5,
                    0xad455a9e36e5e3fe,
                    0x8e3697b3f31a0671,
                    0x0e8452b8927fce78,
                ])),
                Felt::new(BigInteger256([
                    0x4fb2d6f735f4b0a2,
                    0xf375e4170617b1e3,
                    0x2b288dff90a56400,
                    0x1b8c6567ceee4ea1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x703c1d4d11119b10,
                    0x5f12ef809f07ef5d,
                    0xa7ac2beee362caf8,
                    0x33b39a0cdc4ca166,
                ])),
                Felt::new(BigInteger256([
                    0x3d9c128782532982,
                    0xcb0b68f503534df2,
                    0x223fc1a80689b262,
                    0x1ab729ddbfb2d7f0,
                ])),
                Felt::new(BigInteger256([
                    0xca821d7aca6601d1,
                    0xbb232412120c8b12,
                    0xed47451106879c61,
                    0x12fb7f5a3002693e,
                ])),
                Felt::new(BigInteger256([
                    0x47cbe84d716dfa46,
                    0x2573d0545058e718,
                    0x7f8bd19a46babbd6,
                    0x287a09007fdeed98,
                ])),
                Felt::new(BigInteger256([
                    0x2aae36da82837727,
                    0x22dc392061cbe948,
                    0x0184c4bc550fdce5,
                    0x0a13e3f178b8e0bc,
                ])),
                Felt::new(BigInteger256([
                    0xa209c14df281e7b0,
                    0x1df1ff2868c765b7,
                    0x39e48aeed526b7a0,
                    0x0d3aa12a4f0ff301,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd26e13a81ae16904,
                    0x7032e345e4e0aa58,
                    0x3d6d6f22c0ed3a05,
                    0x2405d1c2bb0fecb4,
                ])),
                Felt::new(BigInteger256([
                    0x9bbd96e0c826dcfe,
                    0x22865aedf1254037,
                    0xda0c5fa3b10348e0,
                    0x0ae5f212e73070a7,
                ])),
                Felt::new(BigInteger256([
                    0xd1f84ecbf562fef1,
                    0x20c7ae6d8d11b0a3,
                    0xf0af42896a7dda31,
                    0x2560c02a28fd37b8,
                ])),
                Felt::new(BigInteger256([
                    0x579ff8689334cbfd,
                    0x82460f9459ac4699,
                    0xc30acd6813c64462,
                    0x098174e1371dee80,
                ])),
                Felt::new(BigInteger256([
                    0xfcb1427fa69e7d23,
                    0x18e1afe1f3282bce,
                    0x89854ef37acecc54,
                    0x2cfe989217e1db90,
                ])),
                Felt::new(BigInteger256([
                    0x46d4ca2101e68b83,
                    0xa9cf51e4e147bcfd,
                    0xad7f38a547a98bd3,
                    0x0b9012971de8b135,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8717d200e6a90c11,
                    0xc6f30646355be393,
                    0x3989de4c7e9291e9,
                    0x3520117f601fcff6,
                ])),
                Felt::new(BigInteger256([
                    0x882729ab99bc0082,
                    0xe4886f6838466841,
                    0xdff00110444075c2,
                    0x14a7cf11dd409cf4,
                ])),
                Felt::new(BigInteger256([
                    0xcf8662dc58a3af35,
                    0xa6434c3f57dd800e,
                    0x34588ccc3a8d4100,
                    0x3395d2245a8aaf84,
                ])),
                Felt::new(BigInteger256([
                    0xd1eab3463f50f71c,
                    0xd0605e71e3a93c48,
                    0x5734717295e577b8,
                    0x2cece176df9e7457,
                ])),
                Felt::new(BigInteger256([
                    0x8e7b59c81f39942c,
                    0xc6e82443a1c74a73,
                    0xb10cd81766d00dbd,
                    0x26c86b099408c080,
                ])),
                Felt::new(BigInteger256([
                    0x9d63b95a9ba567b2,
                    0x380702bfa787cc85,
                    0x13d12cbe791f1f17,
                    0x3dc4c9fb09b9ee66,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9e0104e885f3a077,
                    0x8fd75547745862ec,
                    0x9739ecab1c96a305,
                    0x3849a9be06137f24,
                ])),
                Felt::new(BigInteger256([
                    0x382e0d4e08e264fa,
                    0xe78964a114a24dec,
                    0xbce8517dded0bd4e,
                    0x103dab64c60e8c54,
                ])),
                Felt::new(BigInteger256([
                    0xd7a1328eeea1538f,
                    0xdbc9415675c25f3b,
                    0x4e3e17f5d738a1f9,
                    0x16198fc744880a71,
                ])),
                Felt::new(BigInteger256([
                    0x411174d9dfbc1ee2,
                    0xe8c8bf82dfbea747,
                    0x95b09f40446607b8,
                    0x3606809816ca2764,
                ])),
                Felt::new(BigInteger256([
                    0xaaf2e36c129c319f,
                    0x09f542e0e7388a2f,
                    0x296441e1039e7883,
                    0x358e0eefed6bf16b,
                ])),
                Felt::new(BigInteger256([
                    0x408075b238c04fd9,
                    0xad0821422349855e,
                    0x1dc9443d1fbd109e,
                    0x0fcc92902d2429cb,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xf7543a2c518b9da2,
                0x49b60ffe3eda8aad,
                0xaed135abac5f1305,
                0x2fb27506cf425050,
            ]))],
            [Felt::new(BigInteger256([
                0x376373dcf6eec72f,
                0xfffb6caeaf0f967a,
                0x05f1293f9ed45047,
                0x06310081ce46707f,
            ]))],
            [Felt::new(BigInteger256([
                0xb21faa4bb0c43d46,
                0x08b0181ea370a642,
                0x2354682c8e45ddc7,
                0x28d2d2443ef99a6c,
            ]))],
            [Felt::new(BigInteger256([
                0x5a380125d62c7dde,
                0x879ffc711eb0d527,
                0x14e221a440fa5876,
                0x07abcb3b34386524,
            ]))],
            [Felt::new(BigInteger256([
                0x99744916446452fb,
                0x872f1a001138dd5b,
                0xbf9201f961bbe957,
                0x06ad4d984e5449f1,
            ]))],
            [Felt::new(BigInteger256([
                0xce2830653849b3ec,
                0x135ff5585657528e,
                0x41c771d136952877,
                0x1de63179d5f8be6f,
            ]))],
            [Felt::new(BigInteger256([
                0x3423c422a9348e50,
                0xee024f0604d96bbb,
                0x1e155ee596959563,
                0x19a85231c69bfcf5,
            ]))],
            [Felt::new(BigInteger256([
                0x1868f9e162b94e56,
                0x7f90dd780bf1f916,
                0x3259beeaa16f0317,
                0x3338ecb8a4fe43f1,
            ]))],
            [Felt::new(BigInteger256([
                0x14d6e8dc05fedc1a,
                0x49dfcc593081214e,
                0xe9c767372e9d7f24,
                0x29d0368a32939ac3,
            ]))],
            [Felt::new(BigInteger256([
                0xc7960432702a2fff,
                0x2822457269993de3,
                0x47676e7cfcee23ef,
                0x12f8a71b34c7a60e,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
