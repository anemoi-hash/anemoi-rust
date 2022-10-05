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
        // We can output as few as 2 elements while
        // maintaining the targeted security level.
        assert!(k <= NUM_COLUMNS);

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
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
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
                0xb2292e0d2658fd4f,
                0x6f553b94273c57ce,
                0xf2079809161bf530,
                0x1d8bc7a4e35a584e,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xdc3c2e5adec22778,
                    0x1ca31b6bbb9ebafe,
                    0x3b385a5aa3b75e13,
                    0x07d8c5f1f15d8cf5,
                ])),
                Felt::new(BigInteger256([
                    0x2412a3abb02b722a,
                    0x0c7e6a652c1378df,
                    0xed9d256d17a5cc4e,
                    0x390e7289c7f85a1d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb3030f488abea355,
                    0x1800291749c324e5,
                    0x70fd7b16d6737d8d,
                    0x0a3d6be207017b9f,
                ])),
                Felt::new(BigInteger256([
                    0x7e81d4dae19e0033,
                    0x08b02ac74a01c46b,
                    0x37af5760ee73aab6,
                    0x2c328bdc8c6f7917,
                ])),
                Felt::new(BigInteger256([
                    0x95d946fd9fb417ee,
                    0x65630807cce64ea6,
                    0xe7aeebe1db790a9f,
                    0x1e8910126fc8c69e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3b9ac01fa7fda06c,
                    0xb9ff37bc1ad77843,
                    0x53d1ae6ef0a570f5,
                    0x1e4e1cc7914a7692,
                ])),
                Felt::new(BigInteger256([
                    0x2fc1d22300f1f0d3,
                    0x116549389df96b35,
                    0xd81f3933d56736cb,
                    0x2382383acd8f8b33,
                ])),
                Felt::new(BigInteger256([
                    0xede3f12da403828b,
                    0x4f854bb873eda256,
                    0x2ecc1090e17ef7f8,
                    0x0091c31bdf18e2c1,
                ])),
                Felt::new(BigInteger256([
                    0x1c9d33eec1868731,
                    0xdcc3477a35a9aeca,
                    0x9f4a8419bbfe3811,
                    0x39b8df2d98515b47,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7e064f74fc72c65c,
                    0xa710cf2d01ba8391,
                    0x2f171a6b43cc539c,
                    0x0579f48a9465c406,
                ])),
                Felt::new(BigInteger256([
                    0x4f235928e8f5405a,
                    0x8c2355d9806e9f3e,
                    0x89c41431f72681f6,
                    0x0ff5422f0a4c758a,
                ])),
                Felt::new(BigInteger256([
                    0xffe552993a1d1fe7,
                    0x669b717182c240c3,
                    0x52e917ec62880c33,
                    0x01b33a73bbf5d7c2,
                ])),
                Felt::new(BigInteger256([
                    0xea9b766f09fbaf78,
                    0xe94004d074ac186e,
                    0x7cb6f281fad79540,
                    0x26f9e5b52f922892,
                ])),
                Felt::new(BigInteger256([
                    0x202691c7c90c2caf,
                    0xdc62e6eb0c50fdfb,
                    0x9a2960492c25836f,
                    0x1549d57ec2012230,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2bf41aad9d8b6e48,
                    0x7deccc440c161245,
                    0xe23c6acc22c1d1d3,
                    0x01e608d8d630df5b,
                ])),
                Felt::new(BigInteger256([
                    0x5a96aab27883d5c8,
                    0x573e8d5ce0871348,
                    0x0d65117f6e9bcf93,
                    0x02d8b3e2fa787292,
                ])),
                Felt::new(BigInteger256([
                    0x8aa573906498784f,
                    0xb5bb78a0b9d8b752,
                    0x7ee1313022e04ec6,
                    0x0c2a877ec39b2c68,
                ])),
                Felt::new(BigInteger256([
                    0xeba206330499ae5a,
                    0xcd226c0de42298d0,
                    0xe4999f4894eb2bad,
                    0x333740d7a606a692,
                ])),
                Felt::new(BigInteger256([
                    0x9fd429cc0fb00402,
                    0x72cab549e451adad,
                    0xec8202e451be3618,
                    0x01dfd33e7875846d,
                ])),
                Felt::new(BigInteger256([
                    0x51000dfc1075a6fc,
                    0xdc5faaac1ea830d0,
                    0x827c0d206ae7e0c4,
                    0x07d6dd9123633adf,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x5ab926594f8bbcde,
                    0x3f25632292828025,
                    0xeb621245a9af89d2,
                    0x36b05a35cb2581f5,
                ])),
                Felt::new(BigInteger256([
                    0x5f2dee641bf46b2b,
                    0x9b74d4d906f597aa,
                    0xd94ec79ff349f577,
                    0x335cf2e1dbc5cf31,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe8c3d7c12643273b,
                    0x7c0ca2791369237a,
                    0x5ea110adc23d4c8b,
                    0x3e78fa26331bea29,
                ])),
                Felt::new(BigInteger256([
                    0x1ca169eb55a72b24,
                    0x0d3334ef530e8a0b,
                    0x564888ff4a234e9a,
                    0x2bed6ec0a801a19b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04deaad3057dc5c9,
                    0x9620997250b588aa,
                    0xeb57bdd62db5f577,
                    0x13fb6391bcd24741,
                ])),
                Felt::new(BigInteger256([
                    0xb55f8380bd0fae0f,
                    0x4682ba571397a1c3,
                    0x4958db4eafa2a896,
                    0x1d6805039f87e0db,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x63717443dae6a120,
                    0x1b761a281c56a002,
                    0x9f751c3f9465a9d9,
                    0x12222e9e28431ed1,
                ])),
                Felt::new(BigInteger256([
                    0x311fba6b1a675e00,
                    0xac969b3b09f6e315,
                    0xca13459c8fedaa2a,
                    0x2f6cea31135b1435,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb2069a47e0d5ca42,
                    0xb2420c4528477f4b,
                    0x670e9412fa554848,
                    0x28f23e02e6656dd8,
                ])),
                Felt::new(BigInteger256([
                    0xd36be188236dda68,
                    0xa475a6407b9b01ff,
                    0xb68aa7ca319903fb,
                    0x08affdf3634c4d34,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x472c208f68f122e2,
                    0x7aa4ce84fa099dac,
                    0xda92191a692e27f1,
                    0x07919265b76102c1,
                ])),
                Felt::new(BigInteger256([
                    0xb1be51ca17434420,
                    0x0a8e9e78af34e70a,
                    0x602ad1de3a5d9726,
                    0x3d4237d40bf1f1ca,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x46b9de2776738edb,
                    0xf3f0cfb11cc7ace8,
                    0x2613e1edc9c00ede,
                    0x3e88282068adb7ae,
                ])),
                Felt::new(BigInteger256([
                    0x66902cb9ea6b43e0,
                    0x95a3dc7187036168,
                    0x753518d80d46d9cd,
                    0x0ec4275508321509,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6aa7749fd7f13af1,
                    0x34555316c42f125b,
                    0xe7c1c90f97261223,
                    0x36618bbb169d3cc4,
                ])),
                Felt::new(BigInteger256([
                    0x9ea76195372eec34,
                    0x61dbb42bbf46182e,
                    0x84dffa0dc317b6c3,
                    0x368fbcb82c145232,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x03ff129dd4789c10,
                    0xcd8700210dc44661,
                    0xda2777d552f95484,
                    0x088b6b700db8d195,
                ])),
                Felt::new(BigInteger256([
                    0xb09f59674365aaec,
                    0x70871bfce5c68b90,
                    0x63bedf06625ee25d,
                    0x3413dfe10b63fc9a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1db9744c93d30c2d,
                    0x0db81b20ab3b296e,
                    0x29061366831b9562,
                    0x2ceb94563119fa99,
                ])),
                Felt::new(BigInteger256([
                    0x166c3d328df6d075,
                    0x3839bba72a0038d1,
                    0xec515b9b35c12d52,
                    0x1eb00461acc9c57b,
                ])),
            ],
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
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
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
                    0x7fc024f5e4fa0c1d,
                    0x7ee6fced5443bf80,
                    0xe989a7f932da77a8,
                    0x3e733b93233e1793,
                ])),
                Felt::new(BigInteger256([
                    0x4ffeafde905c868c,
                    0x102996a634acbd4a,
                    0x087bb77d1c23731e,
                    0x007d0d05aa907c59,
                ])),
                Felt::new(BigInteger256([
                    0xfd1d849c06db3144,
                    0x86a3997ce1d4c4f1,
                    0xaf7d0ccc1a57007e,
                    0x1db6b81b4699c403,
                ])),
                Felt::new(BigInteger256([
                    0x1007ac33dbaa36ba,
                    0xff4d3c0ff396087d,
                    0xbe4b7fa11b05b70f,
                    0x3e2bad93929d6ad4,
                ])),
                Felt::new(BigInteger256([
                    0xe45a3f56e605a299,
                    0x810d697bb986e490,
                    0x4e479e2a4cfd94be,
                    0x0883b6b471606faa,
                ])),
                Felt::new(BigInteger256([
                    0xb00e59e439cfbcc2,
                    0x46a79e9e53d6001d,
                    0x5cda86593a51bc4c,
                    0x3eb3246e2593fa26,
                ])),
                Felt::new(BigInteger256([
                    0x8859ff5af374a507,
                    0x99796d431526c423,
                    0x8647f718b4e6f64b,
                    0x0d08c27a74090d86,
                ])),
                Felt::new(BigInteger256([
                    0xf7f355c3ef28ff09,
                    0x83f3a3ae5ea885b4,
                    0x2590547068bcd576,
                    0x10f8961260273904,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaf2ca06e22f268d3,
                    0x5540cebaa7f221d7,
                    0x3f56060ec3757a02,
                    0x242d087e7c8b4673,
                ])),
                Felt::new(BigInteger256([
                    0xa63a1afc7b76f19a,
                    0xb5765ff09824e489,
                    0xf2ea437f0bc63080,
                    0x26de17efda3b3dae,
                ])),
                Felt::new(BigInteger256([
                    0x94a9f8aec8a84389,
                    0x26679fd34f953d4d,
                    0x6ad6b24233a96d50,
                    0x17c0556860196e04,
                ])),
                Felt::new(BigInteger256([
                    0xcc4cb1e007c1d92c,
                    0x74488a6a6ccbc124,
                    0x44a615356c2dda27,
                    0x111716ff6886405b,
                ])),
                Felt::new(BigInteger256([
                    0x037f3f7d47b81e01,
                    0x5dcf7bf1e34fc1cd,
                    0x4484c0912fd0d89b,
                    0x0418899b7f42cd91,
                ])),
                Felt::new(BigInteger256([
                    0x7cb78f1146f2d272,
                    0x3a276f4eb904b7cf,
                    0xb2e73f3f947991ae,
                    0x1481ed84bca11237,
                ])),
                Felt::new(BigInteger256([
                    0x4f5687f61305cc88,
                    0x483d9a4ecb83b236,
                    0xfbe268a3d365779c,
                    0x2da7ea43b1d8e0be,
                ])),
                Felt::new(BigInteger256([
                    0xac7b259e9db061e8,
                    0x9ffb668c45370264,
                    0x7eadbb961a5fd90d,
                    0x2b0ce43d4cb18aff,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa177d2e9fbf746f1,
                    0xac20c1b369a963d2,
                    0xc547c7149a7ea523,
                    0x328ebde353285226,
                ])),
                Felt::new(BigInteger256([
                    0x7efcf4f265272569,
                    0x3b7c38c318f05a2f,
                    0x6ab3f2fc1aaabc79,
                    0x075f24718d03b14d,
                ])),
                Felt::new(BigInteger256([
                    0x75550f068830013d,
                    0x55ffa85190575212,
                    0x03bddb2157da8ca0,
                    0x14efe35a86f6022d,
                ])),
                Felt::new(BigInteger256([
                    0xf191ecbb1444c987,
                    0x4160fec0f3999887,
                    0xfb5431b57bd21345,
                    0x1e6953eb680b69c9,
                ])),
                Felt::new(BigInteger256([
                    0xffa1cd0c91de2d32,
                    0xbe75a324ae956a7c,
                    0x39acc5167e2119f3,
                    0x05793244203d386c,
                ])),
                Felt::new(BigInteger256([
                    0x46110a4896ed2181,
                    0x54baadc4a8f9e28d,
                    0x6ac919bfbd3ec8d6,
                    0x2a014d16f055d486,
                ])),
                Felt::new(BigInteger256([
                    0xc49121722d073378,
                    0x749c86f7b81c0354,
                    0xb6a845aeb5867011,
                    0x32cdf06d34699bf7,
                ])),
                Felt::new(BigInteger256([
                    0xedabbaea9598b64f,
                    0xfe09140622d0f08a,
                    0xf7714563e25aabf3,
                    0x25a2430c78ca6a7b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbc819e2e2a268fb9,
                    0x51a200b74d93fd04,
                    0x59c25a203b5949f0,
                    0x317a7b05fbed45c1,
                ])),
                Felt::new(BigInteger256([
                    0xe16f3a3395e4e07d,
                    0xdbf1761f321ed708,
                    0xc19e43a8d5585157,
                    0x141dc84d53663c19,
                ])),
                Felt::new(BigInteger256([
                    0x037b673516058867,
                    0xad36f7bd82c55f00,
                    0x04cb07af540e06c0,
                    0x21fab4ac3d0b2fd3,
                ])),
                Felt::new(BigInteger256([
                    0x398493b2d9d8ecce,
                    0x53f97623533b8e1a,
                    0xb468f0e36801bc07,
                    0x18639789ceb9926b,
                ])),
                Felt::new(BigInteger256([
                    0xfb2b7b69c610833e,
                    0x394934c80746c308,
                    0xaefea5573b483430,
                    0x2acbf71492b7f63b,
                ])),
                Felt::new(BigInteger256([
                    0xfeaf8ecf1a232ac2,
                    0x07e7ae77afa31b80,
                    0xa55d199fb92207d2,
                    0x037198012c3b119a,
                ])),
                Felt::new(BigInteger256([
                    0x37f400a252e3482f,
                    0x797f63b8586a4467,
                    0x2d23a09505b8dfbd,
                    0x291d62a8a2b505ec,
                ])),
                Felt::new(BigInteger256([
                    0x6b65e59ba4e44af3,
                    0x5aa3dedc76c4c798,
                    0xf4601d0565f70c63,
                    0x337200655fc32ce1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf23c3a95a2645791,
                    0x48c48022cb78ac75,
                    0x597ec91140f996bc,
                    0x2999c52ea4ca2a80,
                ])),
                Felt::new(BigInteger256([
                    0xea6663a376ab111a,
                    0xa58dc4d85608003b,
                    0x2d1fffaad39b8113,
                    0x21c1881ad856abb5,
                ])),
                Felt::new(BigInteger256([
                    0x7f0c986414150ae8,
                    0x89a6dd844e56f5c7,
                    0xe844a1fc94fcf595,
                    0x3f59c29efbac0229,
                ])),
                Felt::new(BigInteger256([
                    0x35b30cf83e7d6414,
                    0x9e321663c6cfd044,
                    0x3a2d9f0f23677369,
                    0x10a1c9df8089e117,
                ])),
                Felt::new(BigInteger256([
                    0x6b1357cbd34e24d4,
                    0x004a7774b72d71d7,
                    0xa8942d2de06b03de,
                    0x3cc09bcf1d258063,
                ])),
                Felt::new(BigInteger256([
                    0x7bacff9faa957e43,
                    0xea13e9da442a1200,
                    0x13734620f8903833,
                    0x055f6ab7715c14fa,
                ])),
                Felt::new(BigInteger256([
                    0x271fd4e5d49c6206,
                    0xe255c8664982ae96,
                    0x69034caa429552f4,
                    0x287e9bae36bcbb8e,
                ])),
                Felt::new(BigInteger256([
                    0x09c72ed0b2d3b038,
                    0x584473bbf1277df5,
                    0x77353092572e2b61,
                    0x0fe757b8991ad508,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8acd55b4edfe6099,
                    0x6ad623fc1b0e3f5d,
                    0x45d237804648b4a2,
                    0x12d276202e03732d,
                ])),
                Felt::new(BigInteger256([
                    0x291b097a425a5f74,
                    0x3524b59c041caef6,
                    0x7e2d66adb0743155,
                    0x19ac6df79bd83269,
                ])),
                Felt::new(BigInteger256([
                    0xbbeb0dc526e4d703,
                    0x1038dd5ea8a894a6,
                    0x24e8194b28b36051,
                    0x1286fafb8b3c64f1,
                ])),
                Felt::new(BigInteger256([
                    0xb0a69c2503f78dc3,
                    0xc53d03bf021489c9,
                    0x0d59ea716d44b6d1,
                    0x0ba6d4d9740f859d,
                ])),
                Felt::new(BigInteger256([
                    0x57ab5accc722177f,
                    0x6c3b5bf27a1259aa,
                    0x193abc6f150dca33,
                    0x21f2e795a4b2ab0e,
                ])),
                Felt::new(BigInteger256([
                    0x47d59d8a92c3f407,
                    0x453821aaa43d1776,
                    0xb29bb534011054f7,
                    0x16ce1f83d17d1cab,
                ])),
                Felt::new(BigInteger256([
                    0x334307106c3e6678,
                    0xf01857c77381ceaa,
                    0x569e428083761ac4,
                    0x182a80822df90237,
                ])),
                Felt::new(BigInteger256([
                    0xe9c0dc8d8f83d599,
                    0xb72defb252e8dbe3,
                    0x1ca0c3238ce26b4c,
                    0x16b985c2c0da11fb,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x5505194327e0fb3c,
                    0x05fbb07d3c0fdeab,
                    0x0204092c69fca5a0,
                    0x3abe5103950d04f7,
                ])),
                Felt::new(BigInteger256([
                    0xc868a3c2194afc2c,
                    0xf89846e59c12287e,
                    0xe03909b89e369a1d,
                    0x2db3c564b9bbd6e0,
                ])),
                Felt::new(BigInteger256([
                    0x299eeee25803fdcf,
                    0xb098b3c78e85abf7,
                    0xfb070fa9a6676f5a,
                    0x37cc68c14abaeda1,
                ])),
                Felt::new(BigInteger256([
                    0x0e8b0a9aaea1a1b8,
                    0x82ee94be267360c0,
                    0xa86a67b8001998f2,
                    0x095d2c5fb4258763,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6b0164e9b90ca2ca,
                    0xefe404bdefb0bd56,
                    0xa1f22aa02571ef7f,
                    0x0f589bf37af58994,
                ])),
                Felt::new(BigInteger256([
                    0xe27e4b8ee6c8eff1,
                    0xaac98bc2ec48fcb1,
                    0xd18a017755541176,
                    0x0e96e5b4ccbd32ca,
                ])),
                Felt::new(BigInteger256([
                    0xb5d95dd9eedb8cf5,
                    0x5acc533789860ce7,
                    0x6d585f4dc34477cf,
                    0x0cb9af7d707e2bc3,
                ])),
                Felt::new(BigInteger256([
                    0xc901c4a440688088,
                    0x2e91aa88401737b2,
                    0x3d5dcf38d7f7f2d2,
                    0x2af0752e2db81446,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3971b7839a3acbf7,
                    0x9f82a3411504efeb,
                    0x4244b3db27388646,
                    0x3c98cdec12263ac5,
                ])),
                Felt::new(BigInteger256([
                    0xdccef9455346e5b3,
                    0x92de9e5f6c57a421,
                    0x2c161dd275fd5dd8,
                    0x1f24234eed7b161f,
                ])),
                Felt::new(BigInteger256([
                    0xaab66becaeb6b7eb,
                    0xe33de67c08b373cf,
                    0xfa9301c5280d1e95,
                    0x25110d59084c9750,
                ])),
                Felt::new(BigInteger256([
                    0xf19c5fad1a78d384,
                    0x832110f6308e011c,
                    0x36c544565071fc0d,
                    0x3fd22d1bec9d4e31,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x971e7e821306a350,
                    0xded21f1471cf8104,
                    0x3e062139435e99b1,
                    0x07639c5bc05cc116,
                ])),
                Felt::new(BigInteger256([
                    0xa7c17cab0773dc91,
                    0xbcb7482cdb03b503,
                    0xe23c9875482c5a81,
                    0x35bdd0bb86766fd2,
                ])),
                Felt::new(BigInteger256([
                    0x5b4fea36f3dd1657,
                    0x8374f24c0c536c4c,
                    0x65d9053d16297a7c,
                    0x0626ab60c6661d95,
                ])),
                Felt::new(BigInteger256([
                    0x8548b7ae382085f8,
                    0xb46395d1800776ab,
                    0xc344ed43dd9ba6bb,
                    0x3805be12114bf6b5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x969a755f3b5cd11b,
                    0x570d40f98cb69ae9,
                    0x42312a0714dce025,
                    0x0fb11a6b14bbe874,
                ])),
                Felt::new(BigInteger256([
                    0x6fe9f9c2fe31a286,
                    0xe120889ad21ed070,
                    0x68c49d2740121559,
                    0x0b517f6acde8941e,
                ])),
                Felt::new(BigInteger256([
                    0x3fdbe90d51c53c08,
                    0xfdae3c89b5669934,
                    0xe73e404b6056796e,
                    0x01bea6ada72bef10,
                ])),
                Felt::new(BigInteger256([
                    0x17c76c6d417b63fa,
                    0xd630895399f5f8ee,
                    0x4203408eb375f108,
                    0x0cbf61411ac43389,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0ceb9e1e28b59332,
                    0x8ecf34671cf8baf4,
                    0x3ff9f79777073bb9,
                    0x158655ce3b5aaf12,
                ])),
                Felt::new(BigInteger256([
                    0xd003eb0f1720ee76,
                    0x7366be516c6bdba5,
                    0x5b0a4562b0d21167,
                    0x3a32744ac84950c2,
                ])),
                Felt::new(BigInteger256([
                    0x13fb22a5db8336c8,
                    0x711e8f41695005dd,
                    0xb536884f95e34ede,
                    0x26dfaa1ca4559f63,
                ])),
                Felt::new(BigInteger256([
                    0xab0b64e115179dac,
                    0xc6bfe96aad29b7cb,
                    0xb0524e6f75913652,
                    0x0aac5b5e7b5d4596,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfdcdbe5256491b14,
                    0xddfc5f9fc3a7b603,
                    0x7a50cc1787bec09c,
                    0x0b3094eb4a591119,
                ])),
                Felt::new(BigInteger256([
                    0xe18635fbae3ed65e,
                    0x01f2ef7e15a32736,
                    0x9f44480d82349448,
                    0x0a8d69a99e72a446,
                ])),
                Felt::new(BigInteger256([
                    0x8edebc29e44d8115,
                    0x2eeecedb0e538a7b,
                    0x4e418af01ca318f9,
                    0x010792e3c6b0d503,
                ])),
                Felt::new(BigInteger256([
                    0xf88a2fdc61101e9e,
                    0xe46116a25263f1ea,
                    0x403511463a898876,
                    0x2e523a3800d6f1a7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x89ebcdbf4893766f,
                    0x961e411fbbdaee9b,
                    0xdfa0c1d42240dee8,
                    0x065631aa36d9eec5,
                ])),
                Felt::new(BigInteger256([
                    0xc7ceea4979c3b14e,
                    0xb0b1556caf9fa47e,
                    0x4165d69baa0760c7,
                    0x15156846cd8e98fd,
                ])),
                Felt::new(BigInteger256([
                    0x40987b67fbc3be25,
                    0xe411aa67deb55a36,
                    0xe42486f99863f53b,
                    0x2f0c9cc1583c8e06,
                ])),
                Felt::new(BigInteger256([
                    0xb030443332394ae0,
                    0x78292fc63de8fd7b,
                    0x473de11d3c87f07a,
                    0x04ea6fc8d4a9ac8b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x60c59ad3f58dcafe,
                    0x5994d4384a73144c,
                    0xcaddedcbfb584ff0,
                    0x0097d2e609168d66,
                ])),
                Felt::new(BigInteger256([
                    0xa9cbfb9f5249d13c,
                    0xacd01e40a3e5d3e1,
                    0x402a99878ba28e88,
                    0x1cc5ea19e0937c75,
                ])),
                Felt::new(BigInteger256([
                    0x7602ea67b27e1bee,
                    0x79933736a254cfbd,
                    0xaf916e6f8299f9a0,
                    0x1497208e528f4ac3,
                ])),
                Felt::new(BigInteger256([
                    0xef915a85240a1ecb,
                    0x9acb40803ca3c0d8,
                    0x0b52167f34d7ff67,
                    0x1102f18e5ff24ee1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6d6f4b91f35af0fe,
                    0x3f36f48cb6da29e4,
                    0xdb4754ea331ff6ab,
                    0x03155a7e8670e90d,
                ])),
                Felt::new(BigInteger256([
                    0xf672af5f9459590a,
                    0x7cad1573933f151b,
                    0xbacce443ba9d5926,
                    0x02711119aa54c201,
                ])),
                Felt::new(BigInteger256([
                    0x9a4538952fffec8b,
                    0x5c2cdcac17ad91fc,
                    0x4a62d73af50bacc6,
                    0x1f86aa8644555de6,
                ])),
                Felt::new(BigInteger256([
                    0x0c762f9f53027266,
                    0x49f8450a407d6ce7,
                    0x65d31b3faa275a76,
                    0x28966d53839e4655,
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
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
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
                    0xcc64bb882eec22b9,
                    0x36ca3734c93daaf2,
                    0x6e6aad5e2a9e67d1,
                    0x2760d5299899743d,
                ])),
                Felt::new(BigInteger256([
                    0xe28b32935396603f,
                    0x4e8c5247e0081d31,
                    0x826126b07e1816bf,
                    0x09eca0fe6b982e38,
                ])),
                Felt::new(BigInteger256([
                    0x8c8df043e70d8b0f,
                    0xdde81ede99858139,
                    0xd0ba7a07b17918a3,
                    0x1c227ecfeb9706dd,
                ])),
                Felt::new(BigInteger256([
                    0x85883f73db51befd,
                    0xeae8a8e708c357ab,
                    0x3241d241b08bcdd0,
                    0x249658e936a96f94,
                ])),
                Felt::new(BigInteger256([
                    0x6a12419aba13489b,
                    0x89426c3517211741,
                    0x85c1b7f05ac3c7c3,
                    0x0efa8bd03f48da71,
                ])),
                Felt::new(BigInteger256([
                    0xa142cadd7c61e75c,
                    0xe1216c77eaa9e363,
                    0xac622a87679c5e20,
                    0x36f1ac5b82940c47,
                ])),
                Felt::new(BigInteger256([
                    0xf57119d976405272,
                    0x0150c77e2581b1b0,
                    0xb6673ba8e9b8c7d0,
                    0x2402219f44eb3bd5,
                ])),
                Felt::new(BigInteger256([
                    0xaa0da0c054d4b1b2,
                    0x20cd0e981f4fd1ae,
                    0xb2f00252cb54b7cf,
                    0x020aef731840bf1a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc81fd8ba1f0848b8,
                    0x8db89d294043efe7,
                    0xc5bb2de93eb97999,
                    0x3e3c4ddd267dc2aa,
                ])),
                Felt::new(BigInteger256([
                    0xf6bfd53dc1eaf127,
                    0xe219863db0881a36,
                    0xed5b3baac1b08fa1,
                    0x18eab9e3f4a0bd2f,
                ])),
                Felt::new(BigInteger256([
                    0xef28632582f2ed32,
                    0xc8f613c32ba843c7,
                    0x089fc251a9ec9760,
                    0x31e03f23cd989be7,
                ])),
                Felt::new(BigInteger256([
                    0x46fadfaa480b2910,
                    0x477aed37c9bc5d17,
                    0xf109ce18433ed03a,
                    0x29c01d342b35a98f,
                ])),
                Felt::new(BigInteger256([
                    0xdab3d4bddf5e5e90,
                    0x64f8df2714bce541,
                    0x1dfbcabf9031756b,
                    0x02c3b78b3ecbabbc,
                ])),
                Felt::new(BigInteger256([
                    0x172461bccf288bfe,
                    0xaf428fe74bf5f810,
                    0xb2e4d14d1a28870d,
                    0x04165528955c7382,
                ])),
                Felt::new(BigInteger256([
                    0x691d796bfe79990b,
                    0xa5fdfb02e47e5a3c,
                    0xd7e5614bba42cdfb,
                    0x339c62f02ce9ac14,
                ])),
                Felt::new(BigInteger256([
                    0x1f412052cd7e52bd,
                    0xea41b31b1d5f2f51,
                    0x9a6d55be82b94104,
                    0x2af21f021abb55cb,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x08f5a67266f4d73e,
                    0x5dd4cc0b3a94f8f6,
                    0x7c15c8112444c125,
                    0x15c602f8da9912cf,
                ])),
                Felt::new(BigInteger256([
                    0xa315b6d075f0f659,
                    0x5e193650e6bc1b36,
                    0x1b606e7b2bc28a8c,
                    0x176d52dcadd6adf4,
                ])),
                Felt::new(BigInteger256([
                    0xb6b9091d5b6e1d2f,
                    0xe052bf87dc1c53a9,
                    0xeb28a6007068022d,
                    0x2dc5f1c81a229b8d,
                ])),
                Felt::new(BigInteger256([
                    0x10db2cafe9d34807,
                    0xd58fbca806801909,
                    0xef6e33ef78b434dc,
                    0x1b34476a486a65cc,
                ])),
                Felt::new(BigInteger256([
                    0x1a20363f34fa729b,
                    0xec4abc272b5a0fae,
                    0xd587aaea93c68047,
                    0x32d02fef2e2f4057,
                ])),
                Felt::new(BigInteger256([
                    0xf2299cc0422bcb73,
                    0xea77b1c24d243531,
                    0xc377fb3926508834,
                    0x2396d013f30fc08b,
                ])),
                Felt::new(BigInteger256([
                    0xe8ef3dee485caf2b,
                    0x04612e6bb0125d02,
                    0x0d86191a57c15a66,
                    0x0c431e9c16794fd4,
                ])),
                Felt::new(BigInteger256([
                    0x6b34aa947bfd3dc8,
                    0x3beb47f0aaf4730e,
                    0xeb2b4cf5f87512f8,
                    0x2e082cede317f991,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x51242b347898c482,
                    0x5de9f38bf92ca844,
                    0x5b4c1b5311899dff,
                    0x242608f6a9cf186c,
                ])),
                Felt::new(BigInteger256([
                    0x113c79b03eef6d29,
                    0x4256024d80731df0,
                    0x8b48a5a0fbfb65aa,
                    0x023132a91b3ab5d3,
                ])),
                Felt::new(BigInteger256([
                    0x9a170862624f37a3,
                    0xa92549d08b0c8b74,
                    0xe7006288251f18a7,
                    0x2ec37ac0409e1e57,
                ])),
                Felt::new(BigInteger256([
                    0x3708f4674b9f49d6,
                    0x2501f04f192f7a30,
                    0x6a087189736d4bd9,
                    0x1e42b75b9f9c264f,
                ])),
                Felt::new(BigInteger256([
                    0xf958c3604d3081db,
                    0x1f0e217fe9119611,
                    0x73694d5bcba4919f,
                    0x3d0202c7f1b3ebb8,
                ])),
                Felt::new(BigInteger256([
                    0x0cfbf5211167d505,
                    0x0c467714af12f500,
                    0x847d8014a1431aa7,
                    0x1f858b6912d7714e,
                ])),
                Felt::new(BigInteger256([
                    0x84e1a28be53c6028,
                    0x50b858216dc44677,
                    0x8a5737298cd3888a,
                    0x24d9edbf651e8635,
                ])),
                Felt::new(BigInteger256([
                    0x48216816362722da,
                    0xf235ab34af46f1c4,
                    0x3a2a9d0b112d9ceb,
                    0x0dcdd329f668799f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdd3a302dc1a8f280,
                    0x1f47ab6f0d17e91f,
                    0x5ec9f42eadae774f,
                    0x1f7e5164e51f2c7b,
                ])),
                Felt::new(BigInteger256([
                    0x08df6634177cbddb,
                    0x39d5c83e04883716,
                    0x4f32ac2d2e9c691e,
                    0x2dc2ac576d784d2b,
                ])),
                Felt::new(BigInteger256([
                    0x1e7be5376a144f15,
                    0x4aa928b5a77b87af,
                    0xb917d35a548d7197,
                    0x3fa17cb2968d0999,
                ])),
                Felt::new(BigInteger256([
                    0x320653b8e5ffe842,
                    0x984ef90ac4d70a5e,
                    0xb57b3752eb68559b,
                    0x1ab73669a58f45ad,
                ])),
                Felt::new(BigInteger256([
                    0x2403944d84c420d1,
                    0x3b50b7c8031ba61f,
                    0xafd3ee28d0c77e07,
                    0x38528a5eac13cc10,
                ])),
                Felt::new(BigInteger256([
                    0xcf97121f464381b8,
                    0x28077fd6d82a3f72,
                    0xb847bf1530e78168,
                    0x0a75e75084ef05bd,
                ])),
                Felt::new(BigInteger256([
                    0x5205796ac3ea88e5,
                    0xd352961b14e1f127,
                    0xfac6ce0b6d35add3,
                    0x103e5c3fad0f4721,
                ])),
                Felt::new(BigInteger256([
                    0x13a126b836cab166,
                    0x3e726d8880326f08,
                    0xc97966b92ae553d1,
                    0x3c3dee2386822262,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2375b6d34c968d30,
                    0xf744c41eba5f65d9,
                    0xd99f32799ba2623a,
                    0x345971927b48d13b,
                ])),
                Felt::new(BigInteger256([
                    0x1ec3afd3cbc6c2b8,
                    0x3f1bdbfb7f2be29a,
                    0x956f0d6c2f438658,
                    0x0a6366d1d7e5a0d5,
                ])),
                Felt::new(BigInteger256([
                    0x646a398c960c8cf8,
                    0x97f7d3ae7e09ecb4,
                    0xe23193895acfac4d,
                    0x3c189035a4cab62b,
                ])),
                Felt::new(BigInteger256([
                    0x3360e7d86c0e4a40,
                    0xc62984dda8a12390,
                    0x109cd3401745d586,
                    0x0baa49705a5324c2,
                ])),
                Felt::new(BigInteger256([
                    0x0a6b1194746de063,
                    0x7bfdf4142267fa75,
                    0x99852cb22386522d,
                    0x1236b142f8b1ebc9,
                ])),
                Felt::new(BigInteger256([
                    0x0e61ffdd0d1aa9a8,
                    0x678d77914015f075,
                    0x99b9de7e42fe616e,
                    0x0dba06b3d2e4b7f9,
                ])),
                Felt::new(BigInteger256([
                    0x33e6d71875b93b2a,
                    0xf7ecaf3850d0196b,
                    0xfacc09b4ec5c0db0,
                    0x0ca5457750cfe4d1,
                ])),
                Felt::new(BigInteger256([
                    0xd53cf0ceb8a8356e,
                    0xe65b8398c8e8caef,
                    0x7d1b299a191ff1c9,
                    0x29fac4f672cd3c57,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe576d7387fe4f90a,
                    0x944dcb48c1489186,
                    0xfd0b18d6106414fa,
                    0x328ab9c4dfc7f298,
                ])),
                Felt::new(BigInteger256([
                    0xd6f3ae5cc7ec9de4,
                    0x7b86dba3c285893e,
                    0x88a371709e503310,
                    0x3710f1c46de15e44,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x20dac2c3a7e82fbf,
                    0x4ab057f57936ca3e,
                    0x0f4a89ede8b6674f,
                    0x1c124b70eb73b558,
                ])),
                Felt::new(BigInteger256([
                    0xab80103327317079,
                    0xd95b364b2c603464,
                    0x0ee7d0b02d4c0448,
                    0x39875ae2fa754711,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4afaf28348f183e1,
                    0x6079f0c1146b6a9f,
                    0x3cd7b5a04f45a4dc,
                    0x21a9db451a72d216,
                ])),
                Felt::new(BigInteger256([
                    0x353e28056dbfb936,
                    0xf3b916599398ac23,
                    0x62db6228c66f59e5,
                    0x1ef6506ada186450,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf26e68b906e3b9a7,
                    0x624711607e22ed50,
                    0xa3df26765988142e,
                    0x0d8a47bc86c2deab,
                ])),
                Felt::new(BigInteger256([
                    0x93dd036c3f946288,
                    0x4ed4450251be3293,
                    0xa58185b925c8013d,
                    0x2dc38ecd97c26688,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa6517aeeb0cd66e8,
                    0x942a4e2223037c15,
                    0xb0fd61dcdfebd592,
                    0x33ce244deef69b05,
                ])),
                Felt::new(BigInteger256([
                    0x5083f55cf7a9e0ae,
                    0x0cb665f509a72972,
                    0xe49de1c384c0b95f,
                    0x2d59efe84bbe09d3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe2d8c594ffbf4941,
                    0xc86d0d166f4029db,
                    0xe092bbee90330f3b,
                    0x3defcb6aed223d2e,
                ])),
                Felt::new(BigInteger256([
                    0xdab9664fd3f3f1b8,
                    0xebdd50bfe2ba8744,
                    0xcbb9f21130a78926,
                    0x3dcca4c4044e788e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x45c466b842fae902,
                    0x01a6461f68bc7211,
                    0xdf7829d8a61cd064,
                    0x2f12d037240e2325,
                ])),
                Felt::new(BigInteger256([
                    0x81ee10e384bc3c4d,
                    0x8c171c3eec7ab734,
                    0xf03c9715150d212a,
                    0x21bdcd968d27dc39,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xed76c19c578e6eb3,
                    0xfa696f1179b1178f,
                    0x14c06c4f7efc475a,
                    0x068f7ca35c88f1fb,
                ])),
                Felt::new(BigInteger256([
                    0xf4114cf6635c5872,
                    0x6fa1fbd0a871b973,
                    0x9adfda60e3dee1bb,
                    0x3ce21308618062ab,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x37c761b37834d0fd,
                    0x57a7bc00039720e6,
                    0x9965c28831f58564,
                    0x0bdd2dfa2415dade,
                ])),
                Felt::new(BigInteger256([
                    0x7e6174a9cae9d3fb,
                    0x52ccbdcbfd7aa65c,
                    0xc65d9fa7e23489d3,
                    0x38fb34235b42d248,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb6e59ea3dd14a61a,
                    0x5cbf8099e23d3e77,
                    0x0b1c7147b6ff0ef2,
                    0x0a2bcb2d7f90fb02,
                ])),
                Felt::new(BigInteger256([
                    0x8d25e099d4898161,
                    0x87ed8cc1bbb062e0,
                    0x9c8620de5abb7b05,
                    0x129c686f8f3ba490,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
