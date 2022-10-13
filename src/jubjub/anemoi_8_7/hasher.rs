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
        // We can output as few as 1 element while
        // maintaining the targeted security level.
        assert!(k <= STATE_WIDTH);

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
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
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
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0xd816f7c4ceb7cf60,
                0x17d3a2a8c676bc67,
                0x4c55eca9b7c9ba34,
                0x1c25926f695b7d69,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x5339d9da67d5d7bd,
                    0xd924447f0932f6a2,
                    0x9395909c2f5df594,
                    0x0d8761fab851d493,
                ])),
                Felt::new(BigInteger256([
                    0x25c145928685b872,
                    0x2dbae54a697539cd,
                    0x4bdba56021c4b2bb,
                    0x4b6d6e1e5fe6d90d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe44a6fe0de8e8b6c,
                    0x0a143c250c9a5d3a,
                    0x93edf68a406a715e,
                    0x097d8a257c5c9d85,
                ])),
                Felt::new(BigInteger256([
                    0x924a5bf4a784a8fe,
                    0x6a54a8f0d5519fe9,
                    0x2806f27fe02ff559,
                    0x5a93e237d6f6bf66,
                ])),
                Felt::new(BigInteger256([
                    0x499c70d382e4f20c,
                    0x5c1172771876aa4a,
                    0xa990dab896a6c579,
                    0x47268a4be6d92753,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2b705323fc6f7c3f,
                    0x7f8ef5f80ec2b326,
                    0x00b4aef407007e72,
                    0x640c30d70be298cb,
                ])),
                Felt::new(BigInteger256([
                    0x563682f9c06481fe,
                    0x90bde8b1c9287cea,
                    0x2fa0c0e495c20e1d,
                    0x6c16360cf7670dbb,
                ])),
                Felt::new(BigInteger256([
                    0xc091457ef3d141e8,
                    0x0647b0ac0c2b64e5,
                    0x8f39fde11cca4cc4,
                    0x2e50df008a47fa59,
                ])),
                Felt::new(BigInteger256([
                    0xcf7db1a867267eb0,
                    0x1e2bc85d995da9f5,
                    0x41d854b0dc8dd9af,
                    0x4a9bcd53cbcc2e05,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3e6369699f7b717c,
                    0xbd9331b4005aaeda,
                    0x987da36898dbed06,
                    0x0810b3cd6fa915ae,
                ])),
                Felt::new(BigInteger256([
                    0x5b577956bad5eef2,
                    0x1ecd9a44d7702492,
                    0x18e1af8aad72b467,
                    0x73b2069c13dd23a6,
                ])),
                Felt::new(BigInteger256([
                    0xc2f8983d9d62bda0,
                    0x3837523236cf583a,
                    0xd4bd3722387a87df,
                    0x305517cc0435e55b,
                ])),
                Felt::new(BigInteger256([
                    0xf5e74781b88bfdb9,
                    0x515a942264a7c50f,
                    0xc2970f41116b7e57,
                    0x4f1594534dfa85e1,
                ])),
                Felt::new(BigInteger256([
                    0xb1165890f51fac16,
                    0xe6d1f2eaab0044df,
                    0xa21b87f4b3f6ad0b,
                    0x34a1c502cd12c0a4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcc9015b665281de4,
                    0x94f910e03f06780b,
                    0xd91f039a480b260a,
                    0x501aaf03e78cc50a,
                ])),
                Felt::new(BigInteger256([
                    0x571ddc8beba48008,
                    0x5e2c7562ff1e6df8,
                    0xcba51c7c8c2269a6,
                    0x421f14c57dc71f07,
                ])),
                Felt::new(BigInteger256([
                    0x1c07347a928ca321,
                    0x1c9125e647d65471,
                    0xc4fba2cedef6a995,
                    0x0ddb45f35176b717,
                ])),
                Felt::new(BigInteger256([
                    0xaf3615f5ba8cd7ea,
                    0x12d70e0ca76b3228,
                    0xdef5d0eee7e4db85,
                    0x64bc50d67ea24f0c,
                ])),
                Felt::new(BigInteger256([
                    0x03e1568ce9f358e5,
                    0x145a549c7e545810,
                    0xaec08c47c9cf2b28,
                    0x64b1872bacddf714,
                ])),
                Felt::new(BigInteger256([
                    0xf9b2b862ef6ff653,
                    0x982386417c4f3a80,
                    0x44c89b1a7b16e9ca,
                    0x2c124d9957d4cc03,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x460bda3b8434b00b,
                0x9ab77a34ac7eda52,
                0xba4f5b2d418ef143,
                0x580260fa067c77f0,
            ]))],
            [Felt::new(BigInteger256([
                0x9b14b3e59302a416,
                0x16fed27081e2bf24,
                0x88f78830c62785fb,
                0x6cbddecd535509f5,
            ]))],
            [Felt::new(BigInteger256([
                0x6da42d7a0dc87e66,
                0xa449ed54a1bb0c7c,
                0xf1deea5e3cf35981,
                0x326aeab1c97f2e70,
            ]))],
            [Felt::new(BigInteger256([
                0x93ced670853cbc0a,
                0xfaed57b3be14ca68,
                0x7425057f7fa1c2bd,
                0x0d0573995a48d6bc,
            ]))],
            [Felt::new(BigInteger256([
                0x12bec49671175483,
                0x1dbc7d4c9a03f1a2,
                0xc964c6c96dc172a4,
                0x06d1bf7e0ecc1816,
            ]))],
            [Felt::new(BigInteger256([
                0x72cae091842e4a68,
                0xca83d7e5a8f549bd,
                0x503afbe29d764fd7,
                0x2fda750a50f26c4b,
            ]))],
            [Felt::new(BigInteger256([
                0x4b379992e37e3172,
                0x95fdf8af7b87c412,
                0xee302d9650d29fac,
                0x32b52c31a2b1e6b5,
            ]))],
            [Felt::new(BigInteger256([
                0x47767f10152ff6ad,
                0xa69ea6bb87253ce6,
                0x65eb9a8e2c8616a0,
                0x2a55ff422870b2e0,
            ]))],
            [Felt::new(BigInteger256([
                0x86eaac08166ffd2f,
                0x5ce6a7ba34fc7580,
                0xeb9a1e3189cf65ef,
                0x34277611986d2b56,
            ]))],
            [Felt::new(BigInteger256([
                0x1d23935b80cff6e3,
                0x0acc5729f785a2fa,
                0x2da6da182419e618,
                0x665a6b8c89fa0ccf,
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
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
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
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8ff513517fb9bb02,
                    0xd2883d2600ea8ac4,
                    0xcc3b0b1a0e61345f,
                    0x5f3ef7b847ee9612,
                ])),
                Felt::new(BigInteger256([
                    0xc781945916f7a9aa,
                    0x8e40973b0c39875b,
                    0x13918aeb1920dd1f,
                    0x01f53fb2a88aa5c7,
                ])),
                Felt::new(BigInteger256([
                    0xe24e0ec3fe500971,
                    0x501a4855f82ceefb,
                    0x39de205a275719c0,
                    0x6eadf2805d7ff72e,
                ])),
                Felt::new(BigInteger256([
                    0x5a19e6780d324ba3,
                    0x9522b507fdc71bb0,
                    0xa637daef3512a149,
                    0x25233333db141986,
                ])),
                Felt::new(BigInteger256([
                    0x17a6f8470bfbc8b1,
                    0x3a95a29ec9c4642b,
                    0xd82e8496191679b1,
                    0x18755e612e32f971,
                ])),
                Felt::new(BigInteger256([
                    0x0db0749dcbfaa5d3,
                    0xa92eddf6413c9f5c,
                    0x352943da15e610d0,
                    0x73d04193f6a75334,
                ])),
                Felt::new(BigInteger256([
                    0xf58620b147411887,
                    0x03cf01e0e01b1c8b,
                    0xb25f247857e8ac5a,
                    0x0fc94d1aea53ea35,
                ])),
                Felt::new(BigInteger256([
                    0x9af1f0ed01d4d238,
                    0x0923b877b7e72e59,
                    0xa30784c5043a81e4,
                    0x72258d10972fdc0d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc2e3fa4df0cf5379,
                    0x24c4d9e1c8bc6fb9,
                    0x4424cd2259d0faee,
                    0x6f60acdd221ec005,
                ])),
                Felt::new(BigInteger256([
                    0xc1bb84ecb634263f,
                    0xc1c0ab804452d722,
                    0x6517b151b47d686a,
                    0x485f48c74461d719,
                ])),
                Felt::new(BigInteger256([
                    0xd5c36ab18679a883,
                    0xf9f402af9f62e795,
                    0x4ab0a91e94e83a7d,
                    0x5a65fa8b3bd28d2b,
                ])),
                Felt::new(BigInteger256([
                    0x1d0a9d01183d76ba,
                    0xf361e6a86b463fb3,
                    0x2bb9b88eb0a0ab96,
                    0x5834105d81af8e17,
                ])),
                Felt::new(BigInteger256([
                    0x93b8e2dcf21eb059,
                    0xe5ac4aefd81d83e5,
                    0x654bebebd6c07ab6,
                    0x0a40a544190828ef,
                ])),
                Felt::new(BigInteger256([
                    0x313a9fe3f041bd0f,
                    0xa9be1d1dba02d6ef,
                    0x1515e3db8f5f4f2a,
                    0x54a63b15f110d9b7,
                ])),
                Felt::new(BigInteger256([
                    0xb14b3a67b1023b27,
                    0xdde050a27f63287e,
                    0x78303d0bc89cfc86,
                    0x49369f6628633651,
                ])),
                Felt::new(BigInteger256([
                    0xeb461f7fb3dee7c1,
                    0x5a109dd4bfbaed22,
                    0xf643824c63a761c3,
                    0x12741a1972695997,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x30f29f4e07c2922e,
                    0xac177bf0ca353438,
                    0x7b9f318a7f8d6bca,
                    0x7353d35af5b18cc7,
                ])),
                Felt::new(BigInteger256([
                    0x5aa6d1e732cf4aab,
                    0xe108461e956a63fd,
                    0x3d9bb1d33a9fd6a3,
                    0x4a803c2157c47560,
                ])),
                Felt::new(BigInteger256([
                    0x69b5fb3918fda125,
                    0x58444848450540f4,
                    0xaf411752c4dc5d85,
                    0x4efec438c32c3228,
                ])),
                Felt::new(BigInteger256([
                    0x162b4548f6979606,
                    0xc304bfab573aca41,
                    0xc49e90c664cf2a96,
                    0x2cc03139259f06ae,
                ])),
                Felt::new(BigInteger256([
                    0x888de67fe3982aee,
                    0xdd09817ac09806f8,
                    0xc5eaa695094c11dc,
                    0x61796b29e94d8761,
                ])),
                Felt::new(BigInteger256([
                    0x82598f404cbf359e,
                    0x70846d5049f75525,
                    0x77422d720525749c,
                    0x28a106caa667f264,
                ])),
                Felt::new(BigInteger256([
                    0xcb8a02c08a41cda8,
                    0xa28010d54f7490a9,
                    0x9f73add4dc133dd2,
                    0x6a62a60e508f7072,
                ])),
                Felt::new(BigInteger256([
                    0xe13ad0a1406d2a25,
                    0x174293156239d3bc,
                    0x430a5bc3bf906cf4,
                    0x5fe135d60a30ee95,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6611d4b22cce9257,
                    0x6b9243ab463d4f0b,
                    0x7a2a3defe35ab825,
                    0x33809b99844915c8,
                ])),
                Felt::new(BigInteger256([
                    0xaac9a24817a2da6f,
                    0x96701f6afc62bb24,
                    0x3cc6a41bf45d1b89,
                    0x079250feab38f215,
                ])),
                Felt::new(BigInteger256([
                    0x13974884cc1694ba,
                    0xd63b7e85c278340c,
                    0x51f23883b1e8aaab,
                    0x562aa125253a6432,
                ])),
                Felt::new(BigInteger256([
                    0xaf9a2d00814d8e45,
                    0xc10794f04af4a62e,
                    0xfd8fa40b2bb472ee,
                    0x5f2b2cc5d878fc3a,
                ])),
                Felt::new(BigInteger256([
                    0x8f8cfa42d657a98e,
                    0x9a787f8ee791a4a3,
                    0x549ec0dc1fc8fe04,
                    0x180e499aad43f27e,
                ])),
                Felt::new(BigInteger256([
                    0x8797096b7661dcfc,
                    0xf52e513a6bd2c18c,
                    0x0361736101990383,
                    0x5a81762e65121752,
                ])),
                Felt::new(BigInteger256([
                    0x51d3367530ba0915,
                    0xb162cff469ef5315,
                    0xad49d59aab452466,
                    0x3b3013c2119abe48,
                ])),
                Felt::new(BigInteger256([
                    0x4e9fc640831e967c,
                    0x92fae7b6da35936e,
                    0x15b6ffb2950aa709,
                    0x1c77fef96605015e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7fdf07269eaecade,
                    0x7208a64ac5724092,
                    0xf6dbb02b21ad5ae9,
                    0x27b6a340af1616bb,
                ])),
                Felt::new(BigInteger256([
                    0x3bef13d23dd09939,
                    0xb353ebc351ce3bd7,
                    0xda7b3e597055a9a4,
                    0x24d8c1c652793948,
                ])),
                Felt::new(BigInteger256([
                    0x5ab08ac0f6ce7d6d,
                    0x1172fda82d07c668,
                    0xa3effbe5c2ce68d1,
                    0x5f2509d2b1e6b0bd,
                ])),
                Felt::new(BigInteger256([
                    0x3e14d61151dc3c9c,
                    0xab16d4636a13a6cf,
                    0xdef09683faa59453,
                    0x63834f315d90bb32,
                ])),
                Felt::new(BigInteger256([
                    0x1fd1a328f2f6d1c6,
                    0x7cc8852eb5cf60ec,
                    0xfa56d55516199c48,
                    0x3d2564be1d06ca21,
                ])),
                Felt::new(BigInteger256([
                    0x6339988706317075,
                    0xfea4abda702f391c,
                    0xbea236b8c11c300a,
                    0x5a6342d021fd6f85,
                ])),
                Felt::new(BigInteger256([
                    0xfac514673fa47924,
                    0x1e008161e0e007d3,
                    0xaeda68343f20e8b5,
                    0x28822735d0371bef,
                ])),
                Felt::new(BigInteger256([
                    0xb3462ba5242fb219,
                    0xc92a0c2d321bc015,
                    0xd0697e1c463a3ef6,
                    0x41ca2e96a37a7dc1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd6ecbe5a725d7c49,
                    0x5b5e2abfdcead08b,
                    0x01a8a2682b585dc5,
                    0x4b4ee8a42d99633c,
                ])),
                Felt::new(BigInteger256([
                    0xd22122d8323edfbb,
                    0x20615c310262f4cd,
                    0x4ddf475ef7c56049,
                    0x16e124a6a15de2c3,
                ])),
                Felt::new(BigInteger256([
                    0xdf4cc9ddcd4f879c,
                    0xaaf79db38421de5a,
                    0x0d535a1d3bad07dc,
                    0x038ebd318b8659ae,
                ])),
                Felt::new(BigInteger256([
                    0xef83cf62c014ac90,
                    0x8053eb0693867af3,
                    0x1f7742bf30c7001e,
                    0x26219fc98dfc073b,
                ])),
                Felt::new(BigInteger256([
                    0x289ba766a554d4f6,
                    0x6bec74105af904fa,
                    0xb5c3b04ecff932b0,
                    0x4ce99761894dbda1,
                ])),
                Felt::new(BigInteger256([
                    0xb377b3001d2263fc,
                    0x20f7c9f3616aaaf0,
                    0x53256cf65c95ef61,
                    0x1b8a09eb95491765,
                ])),
                Felt::new(BigInteger256([
                    0xd745ccd344a4d797,
                    0x0cb79b8db386f50b,
                    0xcf9dbc89a7b5d021,
                    0x0c81f4314724594a,
                ])),
                Felt::new(BigInteger256([
                    0x24b402677603b589,
                    0x83afc323bcff00ce,
                    0xa75b01d6b48ccd07,
                    0x5ff4992a6cdc9e06,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x38d03906aa625f4b,
                    0xf35b0584f62294f1,
                    0xa8d29f1a9dbdf196,
                    0x05a59911162612e6,
                ])),
                Felt::new(BigInteger256([
                    0xf6a76c64fddece98,
                    0x775058e51e991d74,
                    0x594c37a26dc371e1,
                    0x60c3aaf679afe166,
                ])),
                Felt::new(BigInteger256([
                    0x864c92739a96d372,
                    0xaad35b8862ee9cd8,
                    0x11b2d7f92d7591d2,
                    0x15d026184399a01f,
                ])),
                Felt::new(BigInteger256([
                    0x1129818c3e2465ee,
                    0x0f76e5055e9ddb5b,
                    0x8cc9d15d87647d4f,
                    0x6b84c2c78eb8469c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x94099442b5673b3b,
                    0x982fd96da6158759,
                    0x40c32d06ed2f3367,
                    0x066f90459445808c,
                ])),
                Felt::new(BigInteger256([
                    0x0c656f5d554274fb,
                    0xa79932d6fcbbe329,
                    0xfb66100cb6cf899b,
                    0x67699bef55d187f9,
                ])),
                Felt::new(BigInteger256([
                    0xc946ceeb13a11d66,
                    0x9f57a65584f79026,
                    0x782bdd82188bc481,
                    0x14dee103c5f158db,
                ])),
                Felt::new(BigInteger256([
                    0x8432776d97869423,
                    0x7055d10596b38149,
                    0x9755e2f887d6a87a,
                    0x4c03ec89364ef4b4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x977b6840af71fda9,
                    0x401927eaa509a6e4,
                    0xb4ec0ce6ccdc01a4,
                    0x1e428522b7bc21f8,
                ])),
                Felt::new(BigInteger256([
                    0x3807e8d0a96071bc,
                    0x188dd313e694f7ec,
                    0x62188267684c2936,
                    0x09596b6e6d410c80,
                ])),
                Felt::new(BigInteger256([
                    0x43504cdb39debf92,
                    0x87a2cb13ed92508a,
                    0x054157e2bc7de771,
                    0x33a70d013d17e54f,
                ])),
                Felt::new(BigInteger256([
                    0xd44ca8227d2c387e,
                    0xeb0bc9abaad34e0c,
                    0xe0291a5d0ec3b0df,
                    0x44fa234c13255f63,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf100891ef6952bb9,
                    0xdba7f35993d6545f,
                    0xec56eb71b1e53ef5,
                    0x47e4bc3db9b75eb3,
                ])),
                Felt::new(BigInteger256([
                    0xa4c30ad5c5945ac6,
                    0xb511b4d1a69ee248,
                    0xa0b4be0055dfce5e,
                    0x60339b38fb19fc1f,
                ])),
                Felt::new(BigInteger256([
                    0x8924472006731eb6,
                    0x5b47a8a17dfed2b3,
                    0x283adb6107625cc2,
                    0x5d8e2dd520d99a93,
                ])),
                Felt::new(BigInteger256([
                    0xe80045ad20006dc4,
                    0xe0ad604d4af99548,
                    0xdaf456a673554616,
                    0x4d0ed10eb1d47363,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6f134dc166d8a72e,
                    0x037403a8e4699f2e,
                    0xecc713232427b733,
                    0x59adf29c4d91f313,
                ])),
                Felt::new(BigInteger256([
                    0x617e637a91c1e7b4,
                    0x83c92f9fd99b58dd,
                    0x72f1dfc5b9b34b66,
                    0x2f176479261fb78a,
                ])),
                Felt::new(BigInteger256([
                    0x07140cb6b7376184,
                    0x3d0c5f31431a177c,
                    0x42d14a1d6ba6368f,
                    0x0cad03aad8e20ad8,
                ])),
                Felt::new(BigInteger256([
                    0x5f6df01f4bb86cac,
                    0x396e1011092d4246,
                    0xfe2be6ca5603592b,
                    0x6227ebc948cc7fc4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x83d3bd3a1009eac4,
                    0x3761f854083dbaca,
                    0x899036bdb230e075,
                    0x5f989be9a01edbd2,
                ])),
                Felt::new(BigInteger256([
                    0xd3961ed27757c0dc,
                    0x969073186127322a,
                    0x140c4daf19b101c8,
                    0x656faf864ba99c3c,
                ])),
                Felt::new(BigInteger256([
                    0xd8c686cb235a99d0,
                    0x6a758faaf8324316,
                    0x38fba660cab1a89d,
                    0x1a9770a50e9ff61d,
                ])),
                Felt::new(BigInteger256([
                    0x926ec1ef43fd1879,
                    0xf3a90435d95ba333,
                    0x69b29a9707bdc650,
                    0x5c5d1af19884c967,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb74c810e60158479,
                    0x76d6d360ec03d461,
                    0x095c5d415d9b781c,
                    0x47b0ffa3daac40a6,
                ])),
                Felt::new(BigInteger256([
                    0x31ddb5150104037e,
                    0x811dc6885f056cc4,
                    0x7d0781f4e99fad6a,
                    0x08eb9576ac3b8af1,
                ])),
                Felt::new(BigInteger256([
                    0x144227d3697c58ba,
                    0x3d4587ca42f97101,
                    0xe919c4c6898afbbd,
                    0x645a8d2e17920960,
                ])),
                Felt::new(BigInteger256([
                    0x07578e7d523f6e71,
                    0x2d087517d2865693,
                    0xae1852f0990ea0b3,
                    0x672c10d910c3e37f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5471fa3fdcb31c78,
                    0xb972de7240d9ecb5,
                    0xed6ce11ddae2212a,
                    0x1b568cef4029894e,
                ])),
                Felt::new(BigInteger256([
                    0x9d570154c0620c72,
                    0xd980cf7a29a531da,
                    0x7ef20b64bbfe07e0,
                    0x189528e9cf170f3f,
                ])),
                Felt::new(BigInteger256([
                    0x3c14d5525f8a29f5,
                    0xacd9aa5055df4132,
                    0x1f1094c59fcb5df6,
                    0x25708c426f0a5737,
                ])),
                Felt::new(BigInteger256([
                    0x3360dc83e50bd452,
                    0x22fcfd6ad9d1495d,
                    0xd6a44881a9cbd7aa,
                    0x1b96c48bc5c57817,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x17b4414c4cebd098,
                    0x0d81dee9ca6171bb,
                    0x256b0beee4098c94,
                    0x70daa49440738652,
                ])),
                Felt::new(BigInteger256([
                    0x139d2418a3091416,
                    0x00066fbfd752e1dc,
                    0x2e388e168e58ee2c,
                    0x52c6ae5338bc769d,
                ])),
                Felt::new(BigInteger256([
                    0xb2fecd0850aabb8a,
                    0xdc43299ac2d054c8,
                    0x5bab2d70dc6c6ffb,
                    0x3c3b6d2f76d8d963,
                ])),
                Felt::new(BigInteger256([
                    0x62dec25d0277284a,
                    0x4d5e46ee5dd245f4,
                    0xda080a6813e963af,
                    0x6e93199f001c67b0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x531bce789a35c7d6,
                    0x5ceba05519991daa,
                    0x20c9c210edd4ef0d,
                    0x0e2b560ed1307aa4,
                ])),
                Felt::new(BigInteger256([
                    0xa87a6a2ff03e95c5,
                    0xdafe964b9f2849e3,
                    0x04c80055e437228a,
                    0x4ae6e64dfda47675,
                ])),
                Felt::new(BigInteger256([
                    0xaf5b08c282928fb4,
                    0xcac3431d4da87747,
                    0x1870f8e5de44b4a4,
                    0x1ee8461608fb34d6,
                ])),
                Felt::new(BigInteger256([
                    0x37acb9332278da75,
                    0xeb37f7106c25cf24,
                    0xd6ca5a82f16f42f7,
                    0x1b3589481311a8bf,
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
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
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
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe1d7005a794a6ef2,
                    0xbd09d6b48fa992e5,
                    0x4496b2545b1f881e,
                    0x51457cab9a574147,
                ])),
                Felt::new(BigInteger256([
                    0xacf4c0a82cd8e5ae,
                    0x95b4ce2a5c2d4992,
                    0x2bd66355df892d40,
                    0x0b74e9a73d4065b8,
                ])),
                Felt::new(BigInteger256([
                    0x96094e765f75b9f0,
                    0xe4387dc4ff07f212,
                    0xdaa945476bd70b16,
                    0x5a44665f97535340,
                ])),
                Felt::new(BigInteger256([
                    0xf6428853d93167c7,
                    0x5353cf2058102eea,
                    0xf9c3aed695bf49f4,
                    0x53b8d8bc23053bdd,
                ])),
                Felt::new(BigInteger256([
                    0x817c5e3f505bf2d2,
                    0xc5788ba494181eae,
                    0x1eb85052eef340de,
                    0x59123a26949a4fa0,
                ])),
                Felt::new(BigInteger256([
                    0xe225f34f72d1f2f5,
                    0x8e285b5529840394,
                    0xbd109027253c8d3c,
                    0x3e2978e0a7c83a86,
                ])),
                Felt::new(BigInteger256([
                    0x6aed76cd7e5ee533,
                    0x5689a3aa8acddd7c,
                    0x551919e18a96c3e9,
                    0x52e2f0e7fcfe85f3,
                ])),
                Felt::new(BigInteger256([
                    0x25dfba7c78d56e94,
                    0x361926e4b547a867,
                    0x09b6d1a6f0dedf8f,
                    0x45896d22d9577885,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1b638ef44be59646,
                    0x7aa45f4107228d16,
                    0x0caecc4e7ff65010,
                    0x0538b30ab46f7528,
                ])),
                Felt::new(BigInteger256([
                    0x791360fc0bb503dc,
                    0x002701bffbf6851f,
                    0x773f344e700d8d1b,
                    0x5313a150ec51df0a,
                ])),
                Felt::new(BigInteger256([
                    0x6ed7a1294315cf13,
                    0xfc37efbcc0ea4e3b,
                    0x9b78d2a842e01441,
                    0x19bbbaa11a9e4e84,
                ])),
                Felt::new(BigInteger256([
                    0xb6eb2cc7fb9c1fca,
                    0x8568b154f8fa8d17,
                    0x287c79eff3d03343,
                    0x391f2fd4b3ea8311,
                ])),
                Felt::new(BigInteger256([
                    0x51cc132ee114e3ec,
                    0x94bd801e4fd5c469,
                    0x9a9480540b5c00c7,
                    0x41cad483a67c65eb,
                ])),
                Felt::new(BigInteger256([
                    0x5d1b1e0cd788e396,
                    0x8961dcb4b79a7978,
                    0xc0dcf32d916a321f,
                    0x6a1c88a72986bc1b,
                ])),
                Felt::new(BigInteger256([
                    0xc033cf42c67d5de4,
                    0x7cce4e3f2fb36d09,
                    0xf7fc928650e37fab,
                    0x3984bb6fd3e25c37,
                ])),
                Felt::new(BigInteger256([
                    0xe9529f899dd19988,
                    0x86c7d8c1da26da15,
                    0x7523562aaf66c234,
                    0x68144c7d0e3ee67a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3b473f7fb5627e50,
                    0x503ffaba817f7db4,
                    0x1d1ed14db722e2ab,
                    0x6065c7bf86e6ae80,
                ])),
                Felt::new(BigInteger256([
                    0x2022fa0c0bb3626a,
                    0xda9f571cc8e9b49c,
                    0xe9765e328f4a5fbc,
                    0x702dbecd86cc7ee8,
                ])),
                Felt::new(BigInteger256([
                    0x60a5ee2c431e4a64,
                    0xa870f4e39adbaf18,
                    0x19da0e6eb552b9c4,
                    0x5f15d47f60e32043,
                ])),
                Felt::new(BigInteger256([
                    0x0c0d56ac07f3485d,
                    0x2a3d6a755e74943b,
                    0x7a82411ccd01b352,
                    0x698a43e622433181,
                ])),
                Felt::new(BigInteger256([
                    0x9dc7c7b553010416,
                    0x2db440f1d106a0b3,
                    0x8829240eca2966d4,
                    0x255675d67ebbac1b,
                ])),
                Felt::new(BigInteger256([
                    0x27f9a1e76efc7fdd,
                    0x4c61bdd0c7b8f630,
                    0x65dd8c1b76ead309,
                    0x56b8c8eaffa726a8,
                ])),
                Felt::new(BigInteger256([
                    0xd905e0a9e3fa9d8f,
                    0xe4cf7a79021b7bc7,
                    0x0cb6a88a2db4ab83,
                    0x5043b73cafd5b23f,
                ])),
                Felt::new(BigInteger256([
                    0x7f97224e06685210,
                    0x1d8b2c26072e9d44,
                    0x0cbedb230f9070ad,
                    0x387acf2dcc421402,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe81a6ec42595ae05,
                    0x838e8278b780d90a,
                    0x6657b2329d11b688,
                    0x70790f7b47dfeff1,
                ])),
                Felt::new(BigInteger256([
                    0x6dcacc9f70292fa7,
                    0x59674ee20632f4bd,
                    0x2807895d9b37c3f0,
                    0x040bb5b40d5a8ee2,
                ])),
                Felt::new(BigInteger256([
                    0x223e9b75be5375f6,
                    0x6e47c8d6f3bb8749,
                    0x7d0c21a530310b57,
                    0x73625076213ea108,
                ])),
                Felt::new(BigInteger256([
                    0x4ebab5d20095bf56,
                    0x6fd61f053adde4cf,
                    0x1ebafdaa2c42461b,
                    0x390302256a253439,
                ])),
                Felt::new(BigInteger256([
                    0x1c6834a7ea849b38,
                    0xed888c224fd72d6b,
                    0x74c5e897f7ff5a9d,
                    0x60c1bf7587575436,
                ])),
                Felt::new(BigInteger256([
                    0x5210de44789d706d,
                    0x1dc6a26beb9743c1,
                    0x643852673a83f602,
                    0x69b3ede0a431c008,
                ])),
                Felt::new(BigInteger256([
                    0x764fd3599db17c76,
                    0x4c3efb264f4210be,
                    0x77f6a2c861c95875,
                    0x4a237a21a3a6fd94,
                ])),
                Felt::new(BigInteger256([
                    0x0fc2718d0c623416,
                    0x176515826eb35bd0,
                    0x56f6ca98f132ae77,
                    0x3011dbbb5f9e8f91,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfafa76c262088357,
                    0x0e59dbeff4c53ba9,
                    0xf74694ea34e0bc13,
                    0x14dc260fcc4883b9,
                ])),
                Felt::new(BigInteger256([
                    0xeceec626f9f2a5b4,
                    0x729256410ab889e7,
                    0x36d5cd322ddac2ad,
                    0x1a4e68d05f2840e4,
                ])),
                Felt::new(BigInteger256([
                    0x918c7d7d0d86b526,
                    0x2f826dffa45681c3,
                    0x7fa9241f9eefb2c1,
                    0x6e87e40155173edd,
                ])),
                Felt::new(BigInteger256([
                    0x97b14b926b1f2499,
                    0x20b034a4529e836c,
                    0x7ae6faaadab2855b,
                    0x391b8c466fcc4fc2,
                ])),
                Felt::new(BigInteger256([
                    0x6a316d0cc7cc8623,
                    0x060f2b8ddd91d82c,
                    0x66be034c9acce430,
                    0x6e464de4f2c2c351,
                ])),
                Felt::new(BigInteger256([
                    0x1e23ba5e63fbc42c,
                    0x95793ca1c427310e,
                    0x05dd429648e61415,
                    0x4f46687d49576ab3,
                ])),
                Felt::new(BigInteger256([
                    0xd85b98d9e4989b72,
                    0x8c99d5ef6e1c58bb,
                    0xcbec87211f91faa5,
                    0x26b3b56a6b22055b,
                ])),
                Felt::new(BigInteger256([
                    0x7646c6e072346f9f,
                    0xab130da1e03f151d,
                    0x67afd85f7d1ac2c7,
                    0x670fd0b851e9d628,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8d3b963e9141ba42,
                    0x3449d168d2d31b06,
                    0x8a23b9416a8c5a1e,
                    0x2fbb79a83b138390,
                ])),
                Felt::new(BigInteger256([
                    0x7567c798d655b4ec,
                    0x3e913818c3e45f7a,
                    0x7c8e20346b06957c,
                    0x31e703df2c99fdb4,
                ])),
                Felt::new(BigInteger256([
                    0xf884f016337acd04,
                    0x07937dbc8b81cad2,
                    0x30ef5d9b21cce333,
                    0x0f3ade445e8c4533,
                ])),
                Felt::new(BigInteger256([
                    0x3358f8fb084bdc17,
                    0xeeb8717986cdbdf4,
                    0x4cb06e865040cb71,
                    0x62a78d1b02daf0c2,
                ])),
                Felt::new(BigInteger256([
                    0xb9ee92bb456f3e8a,
                    0x43d325544e8e00bf,
                    0x8df23b89abbb22f3,
                    0x01e96a20f474c29f,
                ])),
                Felt::new(BigInteger256([
                    0x4edbd5d7d3f5c6d6,
                    0x2f89ba61836ac11b,
                    0x01a3a5784f9e678b,
                    0x1ddd0710a54170d9,
                ])),
                Felt::new(BigInteger256([
                    0x53f913f8e62258a1,
                    0x63bf7d2c77e602b5,
                    0x74953f2ca4ba820e,
                    0x09c788df93b26cb4,
                ])),
                Felt::new(BigInteger256([
                    0x7b55f135a9562404,
                    0x146a827c1d0e986b,
                    0x6db7ac6b11eb9b36,
                    0x1140895341b36e3f,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xc6edb96c80fc6742,
                0xd137faf4d649ce9a,
                0x6d61a80bb6b99a94,
                0x73d08594388a5dc0,
            ]))],
            [Felt::new(BigInteger256([
                0xede849f9b5d161be,
                0xfbb8df9cbe7e1ff3,
                0x187125863abf51f9,
                0x5ace526ebcb9d8ce,
            ]))],
            [Felt::new(BigInteger256([
                0xe72046100fdd6774,
                0x7797ebbb2405e168,
                0xc9352985f6c7eb26,
                0x2c4f798b4b9cf5e3,
            ]))],
            [Felt::new(BigInteger256([
                0x06e820c3e29d12f7,
                0x253369140370e6a7,
                0x29c72b696f390023,
                0x6ada07b434446e3a,
            ]))],
            [Felt::new(BigInteger256([
                0x45aee54fb012a813,
                0x5e61704164508c4d,
                0xf311418adf820158,
                0x236a36540c2e126f,
            ]))],
            [Felt::new(BigInteger256([
                0x03afa86143d1ae00,
                0x33d33767a96d995c,
                0x228f787c3e6456c1,
                0x5db5aad9b1bc892a,
            ]))],
            [Felt::new(BigInteger256([
                0xc3bcad1ae6e6d78b,
                0xa6877acc7da6b3be,
                0xda33afb674ed7f9e,
                0x14944976be2be5d8,
            ]))],
            [Felt::new(BigInteger256([
                0x5c720cabc0fa6cf9,
                0x40551ca1a3c144f4,
                0x3c45292e7f92e63d,
                0x480ad95e9f9fe468,
            ]))],
            [Felt::new(BigInteger256([
                0xb9bce4ab04e42af3,
                0x2bebcc5bd85d12d4,
                0x0fd29872ae790f31,
                0x311c8b2e8a8965f3,
            ]))],
            [Felt::new(BigInteger256([
                0x0e74820cdb27936a,
                0x2c850fba60d46b3c,
                0xc11aba8d0e88e18a,
                0x719321aee188bd08,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
