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
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 48];
        for chunk in bytes.chunks(47) {
            if num_hashed + i < num_elements - 1 {
                buf[..47].copy_from_slice(chunk);
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

    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger384([
                0xf7418fad524a921b,
                0x16c95a6037fca6ea,
                0x383ab74fd0151bf5,
                0xf4c945d923488010,
                0xf9d4bf3cc4a408bd,
                0x0323c4a742e1153c,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x017274df91ea2361,
                    0xab7f97c873b2219e,
                    0x32822eea71ca1121,
                    0xbdc4088ffa8ee781,
                    0x0c10fd8f1b2cc28b,
                    0x0cf0b5148a1f2e37,
                ])),
                Felt::new(BigInteger384([
                    0x62cfa174e8083992,
                    0x55abe15ea6f7b87f,
                    0x04f666c4bf5b08dd,
                    0x130333355d55fe90,
                    0x30dbd6fe0e8ed7ee,
                    0x0c7030589240f3b8,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x69a94c2cf09cbb04,
                    0x06e979077ee2339e,
                    0x0948501c901b61ee,
                    0xc5aad235043d9955,
                    0xf719489332e5337e,
                    0x16597061fe3851a4,
                ])),
                Felt::new(BigInteger384([
                    0x29c562b43b2a607d,
                    0x5931c9af1e05c72e,
                    0x7d3fb0efcb726872,
                    0x1e07864d3ccd3449,
                    0xf679c494272f3ddf,
                    0x076c3b69ca8e964c,
                ])),
                Felt::new(BigInteger384([
                    0x41bc14c97b84d908,
                    0x04de1e38b26389e1,
                    0x5a4c34a0603b238d,
                    0xe1eab8752d47b949,
                    0x3f0e9458e1fdc0f6,
                    0x0b3b1cbb012e11e7,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc807df83e8eb8f0b,
                    0xb4032621cf766a28,
                    0x536e2282910c3673,
                    0xb9ddeeaec75042d7,
                    0xf3d56a7934767c0b,
                    0x0bebff1d0e10a866,
                ])),
                Felt::new(BigInteger384([
                    0x4117f310412fc7c7,
                    0xf8ac41bf6a629671,
                    0xb6d9549f1fb7294b,
                    0x58fd81ba8c476847,
                    0x250a6a1bd85d8e89,
                    0x14b84762099ac4f0,
                ])),
                Felt::new(BigInteger384([
                    0x32086c56bb6924a2,
                    0xbc84b0e0592d5f0b,
                    0x26103db04cf93138,
                    0xa4ff103689695f5d,
                    0x74e21e334f7a79fe,
                    0x18f2c3941c424468,
                ])),
                Felt::new(BigInteger384([
                    0x9d0adc7438265aa8,
                    0xbc337ea76c2c2ff9,
                    0xe973a582de4db27a,
                    0x0d9d1e32ff8787b7,
                    0xdabe631c9541977c,
                    0x04c97f37ca4eb18e,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf12f8af5c9f185df,
                    0x9d1c0a7338e14d4d,
                    0x7b9ad1b46f341b7b,
                    0x94120277dde5d1e8,
                    0xb7fd8a3a70495fb6,
                    0x0d2cec328ece3940,
                ])),
                Felt::new(BigInteger384([
                    0x9b613ecbe8ce2b24,
                    0xd3f46bab38c8ca3f,
                    0x8c3fecf7b3a3c52a,
                    0x52c54bc396bf8aa0,
                    0x195f605d822b5ad2,
                    0x165699c77a58ad73,
                ])),
                Felt::new(BigInteger384([
                    0x2f9af0270111ac92,
                    0xc45fbf98c227fd24,
                    0xf3b526177143ffd4,
                    0x4b4793842345cbf4,
                    0xee8a62c0db01750d,
                    0x095694b8e63d5690,
                ])),
                Felt::new(BigInteger384([
                    0xc904ef92f210be9e,
                    0x613379d8718b4607,
                    0xcb4be6c3334f5377,
                    0xd29f13064e773669,
                    0x4deb59072b5f66b0,
                    0x17260c1372cfaefa,
                ])),
                Felt::new(BigInteger384([
                    0x7fb9d7fe0d1cbb80,
                    0xd66cea1cd8e4b711,
                    0xa357c97e60c6c577,
                    0x3e7474caca7ed2f6,
                    0x832f4109aeac6581,
                    0x0e21dfb44f88a3e6,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x280706c9872784d0,
                    0x0f09e18626616171,
                    0x5056afc93f9e2a08,
                    0x0014b6a3cb5eab74,
                    0x3670bf58646052a0,
                    0x06f065db7d32dfb8,
                ])),
                Felt::new(BigInteger384([
                    0x304dda399cb2b9e2,
                    0xff6244e8acd45400,
                    0x47d8b74871f026a4,
                    0x76384c2b1d45a949,
                    0x1be28ba3ffbd3093,
                    0x10db90d52d0317fd,
                ])),
                Felt::new(BigInteger384([
                    0x7dace85c7ba61b6d,
                    0x9b4cffe17824be7d,
                    0xa4da030060822dfc,
                    0xd0b3775c337e7879,
                    0xc0a398b054e3d271,
                    0x1735ef261e93c0df,
                ])),
                Felt::new(BigInteger384([
                    0x7397cf6a62c5aae5,
                    0xfa220142a3889d31,
                    0x717076a8a39b9beb,
                    0x463ffe7b48f89d5c,
                    0xc8679576229f4dc6,
                    0x1823b5141286203b,
                ])),
                Felt::new(BigInteger384([
                    0xd02c2bf2ec698673,
                    0x8dc5a4f50cffb031,
                    0xbe4d7fedbbcdb894,
                    0x1b2cdc349c22ed70,
                    0x653dcd2ceb10aca1,
                    0x1521b403b132cd9d,
                ])),
                Felt::new(BigInteger384([
                    0x30d8408e410cf9d0,
                    0xbcf295c0dca3d143,
                    0xb2f107678c2b2057,
                    0x39fe1c3e2388e53a,
                    0x8e238a8b9d45fee0,
                    0x05da3e010663c950,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x73fce31ade137ece,
                0x9ea0d65d4fba3f30,
                0x0b7ef0a1cea05f15,
                0xe8ab95e9e4fed0e9,
                0x191046bb7112fa12,
                0x155172f3a4d671b0,
            ]))],
            [Felt::new(BigInteger384([
                0x6a7985476d8727e3,
                0x9d88d86f3b1afe63,
                0x08e3193a62b51432,
                0xa645bec0dbed9e69,
                0x3cab7294377de132,
                0x1544e56b2791e914,
            ]))],
            [Felt::new(BigInteger384([
                0x3b338a5446298bd0,
                0x394d04b576d69641,
                0x7b9e6e7584135ebe,
                0x06fbd11c36faabcd,
                0x12fceb8b5b4e1553,
                0x04bd82bcfdd0a6d5,
            ]))],
            [Felt::new(BigInteger384([
                0x8acc1d20b0f55edb,
                0x50fa26a92c0db2f1,
                0x8c66887df0e7097e,
                0xb945138d3f4499f3,
                0xa4a416ea590885c5,
                0x0c5bc4a2c56a2c00,
            ]))],
            [Felt::new(BigInteger384([
                0x3327d83d68b5b2f8,
                0xe189d1666e42d9f5,
                0x57ee1a676ef80691,
                0x7f77addc67fea081,
                0x0c02ba93d5c75aaa,
                0x0e6767007700008f,
            ]))],
            [Felt::new(BigInteger384([
                0x3ad9085249254bbb,
                0xea7221e9a1bffeb8,
                0xda68a1405d06949e,
                0xfb6b7d743a69d9be,
                0xa5bdd2eff68fa204,
                0x14509b08c99b1e6a,
            ]))],
            [Felt::new(BigInteger384([
                0xe0c6e889fb86c00c,
                0xe436e7febca4f360,
                0x45c91873f6a7c46e,
                0x1937e52b8b81ea3f,
                0x58194f08fcc7fe78,
                0x024cfc6a89f2a96e,
            ]))],
            [Felt::new(BigInteger384([
                0x61b532dd8db23bff,
                0x342c1f133862a819,
                0x142c8eee3b200341,
                0x3d334d8b2ba686e5,
                0x53c39f288ab6ca66,
                0x0d347e4897994d2d,
            ]))],
            [Felt::new(BigInteger384([
                0x2bb402e086afd2c9,
                0xacc3589d1d11d93a,
                0xcf9edfc441810d81,
                0x8b8291f9d9670d91,
                0x162f6462a478deca,
                0x04ea63d02d06a28e,
            ]))],
            [Felt::new(BigInteger384([
                0x5ac83457000b0b06,
                0xe62b2287fce01f84,
                0x764d975c74eab740,
                0x416db41c89b92a37,
                0x2463575b01768ae3,
                0x173c67a8ff5bb1a4,
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
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger384([
                    0xf412deb7f3249a8b,
                    0xd36046d92ee21665,
                    0xe945a91fc087684d,
                    0x9b6b7b0bf94b1ad8,
                    0xc91f41ee69165c6f,
                    0x0449df8cdf54c548,
                ])),
                Felt::new(BigInteger384([
                    0x26f2fffe22ccc398,
                    0xe639e7d2456675af,
                    0xc1a538b10a3cdc6c,
                    0x17ae0b8edbddf8ee,
                    0xd4161e1191628319,
                    0x0539758e86e1f114,
                ])),
                Felt::new(BigInteger384([
                    0xacf43297dc4bbdb0,
                    0xb321b14a79a906f3,
                    0x5f92a7fd9bc3eba9,
                    0x71000bd500647d43,
                    0x6673d3b91345b3c1,
                    0x132e53f6740d463c,
                ])),
                Felt::new(BigInteger384([
                    0xf75c200b663d1977,
                    0x2ce02c4f1bbc3df9,
                    0x53d8c409ce48476e,
                    0x1461b95a9ca65b24,
                    0x911af28721abf7d7,
                    0x09d60a1cbc669226,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x149d68fc6e97ebf4,
                    0x2573ea4b9902fa2a,
                    0x815532b0ba5d71d7,
                    0xb03c2e6b8b95f6f7,
                    0x0d6309dc77215c2a,
                    0x0f0c8b8c56172301,
                ])),
                Felt::new(BigInteger384([
                    0xa69ebf857b54a32a,
                    0x89c8a448eed42dc7,
                    0xd7e9ab0a430de958,
                    0xeb312cfdf9cd17ee,
                    0xe2c44e6c55466e94,
                    0x14f717f4890d1819,
                ])),
                Felt::new(BigInteger384([
                    0x5e02f170d559d3a9,
                    0x51daa4cdf7880f19,
                    0xc10834638498d260,
                    0x029a8ac2d6bc8ca8,
                    0xb402058979abcafa,
                    0x06b3f4a1e3332920,
                ])),
                Felt::new(BigInteger384([
                    0x8340696dee42eb0d,
                    0x761e5bb6d0ae3b44,
                    0xe154bb41394c6481,
                    0x3e0a543a5473b0a8,
                    0x3bf1136ff1181fae,
                    0x17fbd5256092c984,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x0581d09504ad14c6,
                    0x2ed6c1fae454db6c,
                    0x21446b0d26938d8d,
                    0x7c0e32af93b2a03a,
                    0x3ab596ad1348ee65,
                    0x11a798d8fe4c837f,
                ])),
                Felt::new(BigInteger384([
                    0xfcdc17fb6b437e8c,
                    0xdb32129026989ae9,
                    0x23560cb9642a4c5e,
                    0x1b5dee9a49655639,
                    0x93a6c357c05233a5,
                    0x1863f6406b285a6c,
                ])),
                Felt::new(BigInteger384([
                    0x6ca3db7fa83a83fc,
                    0xb0861ed9a4ca53c5,
                    0x97fa3a49c82ccff7,
                    0x790d602f4a26d439,
                    0xe3ce68b3046b95cb,
                    0x01fdea527f4a293e,
                ])),
                Felt::new(BigInteger384([
                    0x77fc5aa5791f7a26,
                    0xa52c2dbc30c7c3b7,
                    0x2838dfe327a03293,
                    0xc029e11869c44f7e,
                    0x6414226202170162,
                    0x12556a19471c657c,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x850899b2f69dd5a9,
                    0x36afd69e65fa1b2d,
                    0x72fa7eaf9c5721cf,
                    0x7fa996a1f2ddf6db,
                    0xd241f74e8aca3112,
                    0x16f9e9d421171b62,
                ])),
                Felt::new(BigInteger384([
                    0x4b19c396a75e393f,
                    0xa3e781fde02dad65,
                    0x4243c6aecf883566,
                    0x19345264b7541531,
                    0x7498bde1a8239a89,
                    0x07bd901afe3eee0a,
                ])),
                Felt::new(BigInteger384([
                    0x9daa17136078e2b1,
                    0xb8415dc41d5591db,
                    0xf58484505f16e5f0,
                    0x8468694087441859,
                    0x73a8baf61482d301,
                    0x14b7735ca8624fae,
                ])),
                Felt::new(BigInteger384([
                    0x35e1a5a67223284f,
                    0xc2ef8c05c0a8d1ab,
                    0xa41c75bfc3ca8e37,
                    0x5b63882627373193,
                    0xda701d580e210ed0,
                    0x0f32b893ec174122,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x26bb5867b1450bc4,
                    0x208e238b9b13d88e,
                    0x47b3d9ea91d777ec,
                    0x7c55bbec221f3960,
                    0x18afd4fa6b518e26,
                    0x0a5a20e345895cb2,
                ])),
                Felt::new(BigInteger384([
                    0xff271a437bdb5966,
                    0x12d1cb65be6e3b9c,
                    0x54bc7d0d4e02591d,
                    0xa5fd4b28a7342fac,
                    0xb8968669de76ebaa,
                    0x0be9e07011855bf6,
                ])),
                Felt::new(BigInteger384([
                    0x06a3925d5b2549e6,
                    0x8b403afe3b7dae6b,
                    0x80f3e3d2ce72708b,
                    0x0e2c8529d749db18,
                    0xa4a3e655b512e5c0,
                    0x0bdb00f685433647,
                ])),
                Felt::new(BigInteger384([
                    0x0988d5bf1c55c111,
                    0xe85782ad1c69be7f,
                    0x5e5ddaf62abeadba,
                    0xc0501e0b9a443db7,
                    0xd8582f02caf3784d,
                    0x075876fe49965e49,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb2be306779316ca9,
                    0xa9f83f9cc174d32f,
                    0x595817e4048c09bc,
                    0xf54a7aa1604994f6,
                    0x06e2a6468c2d2b0e,
                    0x04dc88f531668074,
                ])),
                Felt::new(BigInteger384([
                    0xf56001e7d69113ef,
                    0xe08f10019894bfec,
                    0x57e93dc3ebf4e6f5,
                    0xa6a88b6d6518771b,
                    0xe576595d717be8bd,
                    0x10b50f668c174b57,
                ])),
                Felt::new(BigInteger384([
                    0x0d904d8ff90dbc8f,
                    0x6c71e5d2a20ff41c,
                    0x81e91abd50466f67,
                    0x42d2ac76148b0dc4,
                    0xc9668a402e9371b2,
                    0x17e7ed6ae7860754,
                ])),
                Felt::new(BigInteger384([
                    0xa5916c5f1ff64022,
                    0x3cb6686d74752307,
                    0x8c0e73e21a43acad,
                    0xbd3c8752cdd0aa86,
                    0x8022016f3044def6,
                    0x115419eefd80e448,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0x96994ac2c24f946c,
                    0xd3526f3305b1af47,
                    0x531d0ea4752126f6,
                    0x05436fbfdad6108e,
                    0x30ee4f5d75c39d49,
                    0x197d626cb0bd2c87,
                ])),
                Felt::new(BigInteger384([
                    0x5c1a84d54f91360c,
                    0xfb6940331e38f8e2,
                    0xbe10a6b7a9755a15,
                    0xd5079a5373347c59,
                    0x5d50e02a9a02e391,
                    0x1407c5564759d745,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc3ae83bd930dfacf,
                    0x6ed5aa15a5c0a022,
                    0x2cc61cdbb5d7d4d7,
                    0x727aa37fa687df65,
                    0xa856fe19c837b4e8,
                    0x191c5efd7fe8bf82,
                ])),
                Felt::new(BigInteger384([
                    0x2d1d8eb681ab7efb,
                    0x3e1e8865e10e8dbf,
                    0xa5bad3a6b640a644,
                    0x6fe481d079c7411d,
                    0x7efcb114cee49fd1,
                    0x02dfe0d3f7e91444,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9323d125739908bb,
                    0x38e48a04ae1c038b,
                    0xfad1a995cebbec68,
                    0xa9dd1015cdee1729,
                    0x5ebc2d057e37f6a8,
                    0x00489e140f83dd69,
                ])),
                Felt::new(BigInteger384([
                    0xf5f66a432a73ce52,
                    0x6adb9a4e60f33cbf,
                    0x72937f663b138b39,
                    0x034f95e8453870db,
                    0x330e17e2e9fa3ad4,
                    0x09dd68fb46414dd6,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x821f3406dee5963f,
                    0x0189970c9fbdf009,
                    0xfaecda6046d8e796,
                    0x9658b1940a9be3ba,
                    0x10b88782038cb7dd,
                    0x0c697c426264f44c,
                ])),
                Felt::new(BigInteger384([
                    0x7d174acbd92addbf,
                    0x53179e824254b82d,
                    0xb21fe2220bcc7a71,
                    0x95745094e7dda20f,
                    0xc0de4388b167c43a,
                    0x11c938b2f4707c80,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7c5f312ceaf221f1,
                    0xaf6be4350a5a3f81,
                    0x8ee05163c9582293,
                    0x71b9021681b4b3f2,
                    0x711d0c2d4f357437,
                    0x18c7f954f1a9ef6d,
                ])),
                Felt::new(BigInteger384([
                    0xa2769390a8116d53,
                    0x1b675d2ead84ffa6,
                    0xdeaba51e7dd3817e,
                    0x2214d52436369394,
                    0x00c0930dc570d849,
                    0x0f5625741cd35168,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb9a0425f50f8acb7,
                    0x646a766a5a4e6ced,
                    0x60e51bbb97fe8e60,
                    0x2e8805130c00c18a,
                    0x922a43aeaf81ad0a,
                    0x0171a3456f1fa80c,
                ])),
                Felt::new(BigInteger384([
                    0xf6d84fcd4bdd3106,
                    0x8b0ec448dcdc326c,
                    0x74c416c88324cbe8,
                    0xe8c8373bb944d01a,
                    0xa7079cd01334020a,
                    0x0a9967970da32091,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4570e494b29f1a88,
                    0x55563bc9ffda87ef,
                    0xdefcae6711b90527,
                    0x22952240d06ba988,
                    0x77d33f56956d10da,
                    0x1029cb988899b31b,
                ])),
                Felt::new(BigInteger384([
                    0xc845183f603f6780,
                    0x951e9bd87fa9cc9f,
                    0xafe56ea16e20d5f2,
                    0x666b425510fc2c61,
                    0x2ea9860d18a7f190,
                    0x107d6fafbf2503e3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x5203ecf16b783cdc,
                    0x5a48b18939ad9bcc,
                    0x8665efe200a59c8c,
                    0x8383e13cd292fc39,
                    0x659dc498435bd53e,
                    0x12d703a1689214d5,
                ])),
                Felt::new(BigInteger384([
                    0x9365b655cbd42593,
                    0xecc0abf880d6218a,
                    0x8321b1dc8ed168f4,
                    0xeb3ac43363eb1e4e,
                    0xc3cc33d2766dad59,
                    0x13ef4614fef4facd,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x560fc4af2f2ecef9,
                    0x88f612866ab92eb9,
                    0x4d880029aa08b046,
                    0xe4d13a3f79c742e5,
                    0xc9f7792565e6535d,
                    0x0ac963ebe40947d3,
                ])),
                Felt::new(BigInteger384([
                    0x3bae69ca8d324f6e,
                    0x9a4a25ad8ed5daa3,
                    0xbcc740b201181352,
                    0xe2149a35b50625be,
                    0xfd2034d5852d1a57,
                    0x14a72dd36c88fe81,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6fb739c71f98cabb,
                    0xe045613667db5adc,
                    0xf562d32931622dff,
                    0x0f870b397c5cd1bf,
                    0x7957ca30f153f5d5,
                    0x009004e6418d9d51,
                ])),
                Felt::new(BigInteger384([
                    0xd5219aadacbb7d9a,
                    0x72e02710ae0fcc78,
                    0xe898f926cbdd35ae,
                    0x8a0cd1097d432f31,
                    0xfe0434c0927589bc,
                    0x07fa5eb586e08a49,
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
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger384([
                    0x3767c0e143cfb6cb,
                    0x9a8f7a048f5162c9,
                    0xbfd775640019f926,
                    0xc2234763c58c2415,
                    0xb0ecca4147a188ca,
                    0x167f06189e3b5b01,
                ])),
                Felt::new(BigInteger384([
                    0x5f14f8c1c391160c,
                    0xe88c87e953bc6447,
                    0xba2fd796093577a5,
                    0xc61ee8a0434d9b9c,
                    0x2f406dbc91309034,
                    0x09c7fb6109e45dde,
                ])),
                Felt::new(BigInteger384([
                    0x334c1fc9bf038bf6,
                    0x9547e5e053691569,
                    0xac249b36e449fd25,
                    0x7c194bc2e079747d,
                    0x3e02feb6c064a3bd,
                    0x040028d7cb997b04,
                ])),
                Felt::new(BigInteger384([
                    0x054579015cd15322,
                    0x3d057fa0f90e9771,
                    0x50b1dab48d7b9838,
                    0x4689aa76d77b53ad,
                    0x5e48cf24071cfe08,
                    0x04e3d205a9f4c1d1,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd6c0039895204934,
                    0x9f0b74d4c8e9bc59,
                    0x6853efcb4b1601db,
                    0x01b00bde25ea5705,
                    0x51290ad421569285,
                    0x0531de9eb7d689cf,
                ])),
                Felt::new(BigInteger384([
                    0x6e023d30a9802180,
                    0x185d3851261ccf5c,
                    0x6ba06a73ad10819b,
                    0x3c3eae57a30c63b4,
                    0x42c947f0d9efe7de,
                    0x04d973968b37909a,
                ])),
                Felt::new(BigInteger384([
                    0x1ddb00c25b172f97,
                    0xfa26950bb2a86089,
                    0x768b8aee8d2d9c44,
                    0x4c1f1154ff20369d,
                    0x571dbc2c1a22371f,
                    0x02d8cc2a92c55c36,
                ])),
                Felt::new(BigInteger384([
                    0x59000cb1176dc792,
                    0x54d9f058ed3aa747,
                    0xf0a3c70111fa57b8,
                    0x3f0d7a4549dd9df8,
                    0x550793eccb5678ce,
                    0x07011bd5deac487f,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5493c4a0632d739f,
                    0xc6850244b5f677be,
                    0x688c9d8b261075e7,
                    0x20b2cb42ad6a5007,
                    0x3c9efd9aabe186cf,
                    0x1727f95a19e2849b,
                ])),
                Felt::new(BigInteger384([
                    0x4facfd600ede1156,
                    0x5ddc885188e9af5d,
                    0x03d2ca9c185cf58c,
                    0xfc1988a25799b946,
                    0x6060fe15ced30730,
                    0x0f0636c27577b166,
                ])),
                Felt::new(BigInteger384([
                    0xd4e93e7e59d868a5,
                    0xc805d8ff6286e27a,
                    0xfacf6094e6c60653,
                    0x9e67be2a2cb23259,
                    0x1d1e1267808af81d,
                    0x1109669a8429969d,
                ])),
                Felt::new(BigInteger384([
                    0xd39724e0b8367724,
                    0xab6fc62f7b102454,
                    0xe3d2defb3a435f01,
                    0x9f4f4e55ea53376e,
                    0xd5daaf114a2b1489,
                    0x006fd0c3d4617e8f,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x8e95640bc0eb4baf,
                    0x9422636cbe89f389,
                    0xeaf7fa1255e415a1,
                    0x839b95e29856f777,
                    0xd03ff951dbe63590,
                    0x110e9323638dca84,
                ])),
                Felt::new(BigInteger384([
                    0x115f69a29e0b816a,
                    0x052099e67e386dd8,
                    0x061c6bcc439c37cb,
                    0x3d6b718ec9e25128,
                    0xfffed46c00298e68,
                    0x145a1b47c28a2c47,
                ])),
                Felt::new(BigInteger384([
                    0x21533c5176f8e366,
                    0x80ce55aa6b954255,
                    0xc7994b5858511d07,
                    0x7955fcb38bb672b1,
                    0xc94cf24e36fe7924,
                    0x0ed08f74091fc361,
                ])),
                Felt::new(BigInteger384([
                    0xaa4b0dbba9e10cde,
                    0x74d662053acb33ca,
                    0x98ef4d05aeffa722,
                    0x091b2732771d2a27,
                    0x4eafffe56755f5b1,
                    0x06550d706ffd5b36,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb00bd82b6252aae4,
                    0x3e4736d7df5e9101,
                    0xa6c025215f15ed3c,
                    0x31704556f6d65ac5,
                    0xbf6827e24c27d211,
                    0x161e3e6e01fb194f,
                ])),
                Felt::new(BigInteger384([
                    0x2804380a0deea58a,
                    0x4d8e0aaea26ef4f8,
                    0x15087cd2121539e1,
                    0x4ffbb3cb62d46a4d,
                    0x83f70ef5924f9654,
                    0x11d75da63c9ec26e,
                ])),
                Felt::new(BigInteger384([
                    0xc03ef63b3e922e76,
                    0xb9d15050245696c7,
                    0xd6e47260999a2e23,
                    0x137d4bc695925de1,
                    0x88f0fced59bdd2f7,
                    0x1801072500694239,
                ])),
                Felt::new(BigInteger384([
                    0x268cbc9b51ff9ca4,
                    0x4ebd4db1155c95a8,
                    0x246c131e1ade6a3f,
                    0xf5119d50ae9e0c96,
                    0x6d1b08ee74c12bac,
                    0x06256c62c2a846cf,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x1e6608cc609e8cf2,
                    0x5d493d0375cd0881,
                    0x67bdb8064521ed36,
                    0xc73cbd0b2ba13c70,
                    0x11775150d07ee673,
                    0x0503690b75a8ac2a,
                ])),
                Felt::new(BigInteger384([
                    0x807b9beca3038358,
                    0x8637acfd8432dc5d,
                    0xfb5e15ad0385378e,
                    0xbcb76385bda18cd2,
                    0x93b9b058fb550a0b,
                    0x065ea68066447f27,
                ])),
                Felt::new(BigInteger384([
                    0xb5f77222a59489be,
                    0x223557c3aa32ca23,
                    0x36b0eb539aaadf68,
                    0x094a72f903b06418,
                    0x5d8e3f02bb8afa36,
                    0x00c32146ab217ddf,
                ])),
                Felt::new(BigInteger384([
                    0x041f43ca86408c16,
                    0x063acc6e18fea6ea,
                    0x5452b072d33d59dc,
                    0x2af27a581527435b,
                    0x98e795dec122370b,
                    0x1652adfd11e5a53d,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x38b4cf9811e11fcd,
                0xb00faf677296a82a,
                0xa9fce2bb27e58ae8,
                0x75d3be8e5a857a28,
                0x432387d1cc7ad403,
                0x138415d8be971d32,
            ]))],
            [Felt::new(BigInteger384([
                0x36cd127414b9cf1f,
                0x8e48327cd57b2de2,
                0x6b501de1756784f7,
                0x7de7d9cb2cca0dc3,
                0xdc38077853d0a7e2,
                0x01fb2de73e51ed2c,
            ]))],
            [Felt::new(BigInteger384([
                0x891a3b689e0cd70d,
                0xa3c024530f0f404b,
                0x6d6528fc09cf77a1,
                0xad2ca5fe13268805,
                0x91ca44e86832317c,
                0x0a26070f55c52b3f,
            ]))],
            [Felt::new(BigInteger384([
                0x45377ed2b810c953,
                0x35f5359030bea837,
                0x45dbe9e15bf46be3,
                0xc755b6a3fef4730b,
                0x867b235471a8cf40,
                0x0431a30b1d558a32,
            ]))],
            [Felt::new(BigInteger384([
                0xfcfc8340be73c9a7,
                0xbd7df4b7c190d617,
                0xce5157eccc19b296,
                0x36e43b134ee075fb,
                0x35b686b0fe71c6aa,
                0x052911d35fceba34,
            ]))],
            [Felt::new(BigInteger384([
                0xcf9f1bcd06dc3958,
                0xdd0c4ab65fbd9347,
                0x00298ef82f35ed41,
                0x227d0a362b7ccb7c,
                0xe2326b5480544f58,
                0x06eabab1fcd5a936,
            ]))],
            [Felt::new(BigInteger384([
                0x9811ad3b71e331f7,
                0x4470359aa763a7af,
                0x7e02a7de82a62b43,
                0xe01d5b578f932762,
                0x945b424863d480c6,
                0x040f2c6f97166fed,
            ]))],
            [Felt::new(BigInteger384([
                0x32fa093af2093fc9,
                0x002f7831d137e88f,
                0x80471ffd7253422c,
                0x68e54239a9b79333,
                0x91f805841d033946,
                0x0dbcc78edf360907,
            ]))],
            [Felt::new(BigInteger384([
                0xc71925405465f5a4,
                0x35f8ae46fbeb9bef,
                0xc40da735af136b5a,
                0x51631a2c9279a1f4,
                0xb3c04d28ccbb02bd,
                0x08cedd1647796d41,
            ]))],
            [Felt::new(BigInteger384([
                0xcdc6080966fdfab3,
                0xb2c12f01fe0f62eb,
                0xa4c390bf3994d66c,
                0xb24222bbbab67a61,
                0xaa43cc10f68c96ab,
                0x12c8e6c0e6170fe0,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
