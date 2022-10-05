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
        // to the whole state. We then apply a final Anemoi permutation
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
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
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
                0xdf5ea8c39ee73fa9,
                0xf5704e3365af7e5a,
                0x3dae4e287355a6dc,
                0x1ddf19f8c8ceb45b,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x19a7146117fbad11,
                    0x299dad0278070052,
                    0xcda3bc33e7c7b77b,
                    0x05657c13093bf6d1,
                ])),
                Felt::new(BigInteger256([
                    0x4f96290ef887dac5,
                    0x423fc3fb60729e81,
                    0xadd552ecc9660202,
                    0x0de7a9b7c07a8df4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x34320ba4a1cf87c0,
                    0xf3c9140498a177f9,
                    0x6efc39315cfb0861,
                    0x17f58fccc0963aa8,
                ])),
                Felt::new(BigInteger256([
                    0xf3defd798182e24b,
                    0xab45dc1e6f77847b,
                    0x269bb2a57c3cba04,
                    0x289c1a51f43d20e4,
                ])),
                Felt::new(BigInteger256([
                    0x9bb20787f458092d,
                    0xb0432654edb70339,
                    0x8bfce9bc0f6c4e1c,
                    0x1882ba60c69ca124,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6a9770ffa4460e37,
                    0x2cba8b7e0fb56ced,
                    0x2c8d7acd9c8a65bb,
                    0x2d3254270386eb29,
                ])),
                Felt::new(BigInteger256([
                    0xfa6c231110d2b848,
                    0x0bfc5d4af250d4fa,
                    0xa285a99b74204fed,
                    0x18ad916dddaf2990,
                ])),
                Felt::new(BigInteger256([
                    0xb6790c3d94794a76,
                    0xf429353caa4cebac,
                    0x1cc5ef9d89e50cbb,
                    0x127315689cd6ed1d,
                ])),
                Felt::new(BigInteger256([
                    0xd109a52887d8a274,
                    0x3d01af514119c182,
                    0x8e4692dc1e25da37,
                    0x0c3128ce148515e6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd7c86e8bb448c7d8,
                    0x58e403d8281ac36a,
                    0x45f5f260d76d9545,
                    0x2749db2aab852013,
                ])),
                Felt::new(BigInteger256([
                    0xd430ea6066771fa1,
                    0xe782be12cc0e6d6b,
                    0x50f299b785e22e84,
                    0x0e57916746ac82bd,
                ])),
                Felt::new(BigInteger256([
                    0xc4dca7cf8ae3d911,
                    0xca6bf5e467782f50,
                    0x5b5607b2b57e5dc3,
                    0x14051d15178257fb,
                ])),
                Felt::new(BigInteger256([
                    0x46a9f82ef5901ede,
                    0x55dcc705f83f0521,
                    0xee948bc9caa7a868,
                    0x0f7c47333bd1c03e,
                ])),
                Felt::new(BigInteger256([
                    0x2981c7ce0d39d983,
                    0x60f16f34fb6be889,
                    0x3a84ed3959e503ce,
                    0x106b2b03d2839ba0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1d07acb40dbada03,
                    0x109af1c648851b2d,
                    0x7c22344913d5bd70,
                    0x2fc7bcea3db75006,
                ])),
                Felt::new(BigInteger256([
                    0x5e9669f425b56ce3,
                    0x3221ca0bc10fc34d,
                    0x46be10b6a8223c67,
                    0x1c1ef478acea5292,
                ])),
                Felt::new(BigInteger256([
                    0x02943e5ef8d2efce,
                    0x2e91495bc2c15be9,
                    0x9f8e8246163e7a9e,
                    0x143b1067dbad9d12,
                ])),
                Felt::new(BigInteger256([
                    0x393ab081127520f4,
                    0x279c2d2f264d446a,
                    0x8c35bd87d5e432e0,
                    0x2ea833e5e4787bde,
                ])),
                Felt::new(BigInteger256([
                    0x54a37636474d6ba1,
                    0xf4242658997fb70e,
                    0x62936d4c81934764,
                    0x0d6e54d5f6dfcf55,
                ])),
                Felt::new(BigInteger256([
                    0xe4be319a34c6e144,
                    0xd9fa7ec0e24fd5a7,
                    0xce409555ed214daf,
                    0x25770581812bca8f,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe1a3f29f0e80c240,
                    0xbbca28fddf1260aa,
                    0x8e2c9ae391b67a91,
                    0x1c58bcbb38d38ccc,
                ])),
                Felt::new(BigInteger256([
                    0xb98396e53fa0bc9b,
                    0xc62d114e6ee714a8,
                    0xa01fb980637a719b,
                    0x20e197044a17657e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xaf305544554bc991,
                    0x9501c8408f64a473,
                    0x32ea60c898998ae1,
                    0x192b2c0c1044be93,
                ])),
                Felt::new(BigInteger256([
                    0x624641afe187e206,
                    0x9b7cbe64fcddd448,
                    0x5d0c71825306be6b,
                    0x190ab93ec49bca97,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x834672d60c14953d,
                    0xb934d7850a5c528b,
                    0x9031da76207ff3aa,
                    0x1b44db8837579983,
                ])),
                Felt::new(BigInteger256([
                    0xa84ce3037769248e,
                    0xdfe7cfb26708a596,
                    0xa6d7d8b83f42fcf5,
                    0x0a36c2f447b1fac1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xab29320003bf832a,
                    0x43e4308dc33365fe,
                    0x71bd45c1b7387092,
                    0x029d85b0302a8974,
                ])),
                Felt::new(BigInteger256([
                    0x5a03bf6e1d8f1193,
                    0x57c40a6f9cbcbc0d,
                    0x5f9beac3831e15e7,
                    0x1619566844415a21,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xff907e0eb5928181,
                    0x989d9c976d3facdf,
                    0x62f6e43e9e5b7305,
                    0x2a74160484bec308,
                ])),
                Felt::new(BigInteger256([
                    0x9220a78e079d0fac,
                    0xc4e39ffd65e3f0ba,
                    0x19ab343bc2f6cf0e,
                    0x2d1643747d2e8818,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf50b050bfda833ce,
                    0x34c36485d29c4fba,
                    0x1a03eb65c5be1ae0,
                    0x2e01a9227e7df2a8,
                ])),
                Felt::new(BigInteger256([
                    0xca176f12c12157e3,
                    0x2ea13fe080e83321,
                    0xa3f8d09d2740deda,
                    0x2dd5a5a37344caf0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb7faf4d1987824d9,
                    0x1ba718b6e85f1c4c,
                    0x03d795656f5f38f3,
                    0x0e6ea808ecab8482,
                ])),
                Felt::new(BigInteger256([
                    0x3e51e0ee6b7c40b7,
                    0x63077672dea5f0fa,
                    0xba91f63f3f34982b,
                    0x0be4faa822c3c81c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x91b900cd7ae57d14,
                    0xf6ce11367d79f49f,
                    0x46fb075169bed745,
                    0x0ca0c1ed4f21aec0,
                ])),
                Felt::new(BigInteger256([
                    0x01df46124d7630ee,
                    0x171eb704b59d9445,
                    0x956716b3d83d06b6,
                    0x23a50b476a85c50a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcc71c3da1d6cac4a,
                    0x6f53cfba1a04ebdb,
                    0x1b00c812f1f7198e,
                    0x227980177cf760d2,
                ])),
                Felt::new(BigInteger256([
                    0x3652fda8db2f0c5c,
                    0x01247fcc68959642,
                    0xc47f187cf6093e3c,
                    0x090cc5249dbf08e8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x14e0931e04d16a83,
                    0x042a412c4c261795,
                    0x43a1281d4e09d0da,
                    0x115be3c36b6df6be,
                ])),
                Felt::new(BigInteger256([
                    0x5a4ea94ab2fefde8,
                    0xf9e1266e0f506100,
                    0x733625cf43420b78,
                    0x0aa514a5e8c3c58c,
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
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
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
                    0x7343415bcb0b1984,
                    0x27dbc17ac53b5325,
                    0xe65d41123f7ff940,
                    0x17ca6372c0dc03be,
                ])),
                Felt::new(BigInteger256([
                    0xcb2a0cd429940570,
                    0x1f795a7c88d3650b,
                    0xf5fa49329870b9a3,
                    0x27eb6de46a291f83,
                ])),
                Felt::new(BigInteger256([
                    0x0fc637424fa36244,
                    0x8558c26e32d8d107,
                    0xc4612d615b236a48,
                    0x243c3d1f86eff023,
                ])),
                Felt::new(BigInteger256([
                    0x3b26be9aa95b436c,
                    0xf44e503386df4748,
                    0xbefd799758bb0fa0,
                    0x0682b8ca49c3a1c5,
                ])),
                Felt::new(BigInteger256([
                    0x75abcd28885066d9,
                    0x1f6528ca12fd41f2,
                    0xf2f714cacf3e916d,
                    0x26b9a2e4e2d37132,
                ])),
                Felt::new(BigInteger256([
                    0x3804fb86a9e62ad9,
                    0x9b69e4871a347b52,
                    0xa4b5055dcd371128,
                    0x2cea4d70859abfbe,
                ])),
                Felt::new(BigInteger256([
                    0xc661ba1496a75e20,
                    0x0c116f5356a87132,
                    0x5477ca9b7d43d3f5,
                    0x051ab233e043d670,
                ])),
                Felt::new(BigInteger256([
                    0x851ff2402633ba9a,
                    0xa12887c8fee497c1,
                    0x46619299d4737db0,
                    0x108f122e80957da7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x75d5514f59ac9056,
                    0xad17ee9ae7bdf411,
                    0x7a29d022c8e41dd0,
                    0x1ea28dfafbc5f0bb,
                ])),
                Felt::new(BigInteger256([
                    0xad7721c14ef1c618,
                    0xfe600db41d5f9abc,
                    0x5401a09ce19edbe0,
                    0x018416a44eeadf39,
                ])),
                Felt::new(BigInteger256([
                    0x9b595fad6363660b,
                    0xb29c62275ea98090,
                    0xaa7753c6ccf2d4a7,
                    0x2a0f89350926305f,
                ])),
                Felt::new(BigInteger256([
                    0x6896e1acb62cdb48,
                    0xb06b0ca03cf89916,
                    0x4244a90e37229ff8,
                    0x08452df04054a202,
                ])),
                Felt::new(BigInteger256([
                    0x118222e4922f2d40,
                    0x6ddb9d9ab36e2266,
                    0x9176d0bec7d56ef5,
                    0x11ae9ddb059de704,
                ])),
                Felt::new(BigInteger256([
                    0xffe0b4c2d87f2ecf,
                    0x099955b3501fcaa2,
                    0xa7ffac44ac395458,
                    0x2e5b561a9a84978d,
                ])),
                Felt::new(BigInteger256([
                    0x1c95bf7f74f6131a,
                    0xf743fd6df7b499e0,
                    0x9e7cfcb87919a273,
                    0x24be325c8b59dca8,
                ])),
                Felt::new(BigInteger256([
                    0x7e4617f5a3656d79,
                    0xfd8a8ea6cd365ad9,
                    0x6291aa86969da1f9,
                    0x2aa46b713fe77c7f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2fa3a7209b823347,
                    0xe02756c2746f475a,
                    0x87a9d0c2249a4c54,
                    0x10693d7f7396c584,
                ])),
                Felt::new(BigInteger256([
                    0x99714c2ad678249f,
                    0x6d9825fb49a4465e,
                    0xd406fa5b58b8ed1b,
                    0x0bf844ab20775f7d,
                ])),
                Felt::new(BigInteger256([
                    0x1c6109c50097415a,
                    0x91048820ac2d1030,
                    0x6c59d62b563e34ae,
                    0x16dc5130b2aa1449,
                ])),
                Felt::new(BigInteger256([
                    0x582d3a066dc7f66a,
                    0xe8263f2f8d0ddf0b,
                    0x593fa9de1516aa09,
                    0x1f23c059aa954548,
                ])),
                Felt::new(BigInteger256([
                    0xc81326e4f815f48a,
                    0x6f620d33b40e36df,
                    0x120e59f2afd8bac9,
                    0x20d276109157a326,
                ])),
                Felt::new(BigInteger256([
                    0x55d19c0a2dfb0572,
                    0x9853eb298782fc5f,
                    0x5df82f5c61883c06,
                    0x23da6edad6ff16b5,
                ])),
                Felt::new(BigInteger256([
                    0x455fb2191ed8a9cc,
                    0x50df2b37922d3805,
                    0x2e0c628a2582c9ec,
                    0x07b7e5265ac354ca,
                ])),
                Felt::new(BigInteger256([
                    0x7ec7afb99ae28b4a,
                    0x20bcb4f343c7c6dd,
                    0x79f4deea917c7a36,
                    0x19f1b0a3844ec8ef,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x54566436616bab08,
                    0x7b6bac268f719cfb,
                    0x95e48d50343ecff9,
                    0x26ccd93293707637,
                ])),
                Felt::new(BigInteger256([
                    0xc382965ee89ef3de,
                    0x6e05fda3222220f6,
                    0x0df20f4cf8b1d2af,
                    0x10352f69e4dda93b,
                ])),
                Felt::new(BigInteger256([
                    0x112cbc8b6dc2d63b,
                    0xfb7dcac30fc72a95,
                    0x976c6af646ca12a3,
                    0x13d45d75c7cecb62,
                ])),
                Felt::new(BigInteger256([
                    0xcbe22d7dd57b1781,
                    0x4664299529d45a8b,
                    0xc268b6f82197828e,
                    0x0f49a47e9ee84407,
                ])),
                Felt::new(BigInteger256([
                    0x1a511b806b71acfc,
                    0x7724b685578d05eb,
                    0xfb0b57b22325991d,
                    0x168fdaae3d354755,
                ])),
                Felt::new(BigInteger256([
                    0x3aae317893753b24,
                    0xd80aaa42c668ae53,
                    0xed51bb09aea2767b,
                    0x25bd1a0c7105f120,
                ])),
                Felt::new(BigInteger256([
                    0x8c666a823df3b516,
                    0x1460a152f20b4812,
                    0x4ce82e4dc4045b22,
                    0x1dac9cddf80a5581,
                ])),
                Felt::new(BigInteger256([
                    0x2590ce89e8ad3cca,
                    0x486692d4143899af,
                    0xed9d0c6309c1e16f,
                    0x1fa7b6d522104b06,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa5c0873edebb44e6,
                    0xee3f6a89371258e1,
                    0xa777df3af13ba9a9,
                    0x148f8061d447243e,
                ])),
                Felt::new(BigInteger256([
                    0x9731afca4e3488ca,
                    0xf81f6d14d32f885b,
                    0x684781d5226e4e2e,
                    0x2f8157c14ea79cbc,
                ])),
                Felt::new(BigInteger256([
                    0x62243d79e0a88a59,
                    0xe981e54750882e74,
                    0x1401f1d0092addc5,
                    0x131eba15e79ef2c2,
                ])),
                Felt::new(BigInteger256([
                    0x50af4e00265b9010,
                    0x3c600c5000a1d1f8,
                    0x55785844e1dbf6c8,
                    0x070c72ea5936df61,
                ])),
                Felt::new(BigInteger256([
                    0xc4fd4b7055d6fccb,
                    0x6a100bb823193eb7,
                    0xcd36080a9a91e782,
                    0x252b996acb5c2ab2,
                ])),
                Felt::new(BigInteger256([
                    0x1c61f068bbcb327f,
                    0xbbfdc1865b3bc5a0,
                    0xde636751c7b39f08,
                    0x0b92467069f83a79,
                ])),
                Felt::new(BigInteger256([
                    0x94ce506e497be713,
                    0xc78a4583bf7e2e01,
                    0xa90d2d73037b1b57,
                    0x0d1db6e2d5b8138b,
                ])),
                Felt::new(BigInteger256([
                    0x249bf306367df8ab,
                    0xcd1703264c12a985,
                    0xba39ac5c8555ec61,
                    0x2c42c70448e44fa0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x66788aae2a9e698f,
                    0x89de13655fcc519e,
                    0x38bdf8e45a725c4c,
                    0x057612c584846672,
                ])),
                Felt::new(BigInteger256([
                    0xadd69c66a11f664a,
                    0x601946e6055df97b,
                    0x0fe9653797e311d0,
                    0x2723c608c169b4b7,
                ])),
                Felt::new(BigInteger256([
                    0xa96c707c16cb84d9,
                    0x3384c92b172e4424,
                    0x7af4cad8536fe4b8,
                    0x0e5a1d8b5c958f59,
                ])),
                Felt::new(BigInteger256([
                    0x8272826385b938be,
                    0x73a9377d024c8da4,
                    0x3462ba1493183b76,
                    0x1feaeec31c55a3fd,
                ])),
                Felt::new(BigInteger256([
                    0x37f840785fb94ef9,
                    0x57dc73a3931b73d6,
                    0x50056355722afe00,
                    0x294da164c245628d,
                ])),
                Felt::new(BigInteger256([
                    0x072b21783a47ea6c,
                    0x09832e29299604f0,
                    0xb732586477f75924,
                    0x08a8dad301c9013d,
                ])),
                Felt::new(BigInteger256([
                    0x1ded6a4d31e70200,
                    0xcafe0c31884201de,
                    0xfedfb9f32602bff0,
                    0x023eb366462c1ca1,
                ])),
                Felt::new(BigInteger256([
                    0xd2b290ea463a082b,
                    0xb956a66cf326d8c0,
                    0x69a77778c028ca59,
                    0x0f19823db9a76c04,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xa6f563fef5ab51ab,
                    0xc7ffdad7132f19c0,
                    0xbd2c76c47297eab5,
                    0x0480e3a7aaa6151f,
                ])),
                Felt::new(BigInteger256([
                    0xfddb373006da51d9,
                    0x638cee46a0677cdb,
                    0x8707c20500bea005,
                    0x08be7a8ddf185c28,
                ])),
                Felt::new(BigInteger256([
                    0x374cef0210b01e09,
                    0x1e83d2da61b15a79,
                    0xba0870a2ae2d486f,
                    0x26a25a1d3fa4fda7,
                ])),
                Felt::new(BigInteger256([
                    0x000209bd6fd9f326,
                    0xdb37c1a98238cb6d,
                    0x146927b0a8c796a9,
                    0x061a6f89824bab8c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8c62969330d17fe3,
                    0xd5470bbd2abcaeac,
                    0xf7c303a65f35859a,
                    0x2dc87727758f9b9c,
                ])),
                Felt::new(BigInteger256([
                    0x2764381bc2a12783,
                    0x9dd1b6c20e82b3f7,
                    0x43146f6f08163838,
                    0x0d8b023baf00395d,
                ])),
                Felt::new(BigInteger256([
                    0x9cd42dd0e554f3fe,
                    0x8d8109e30f4a9a0e,
                    0xb74352c5a63426c7,
                    0x17678aaf43f248d7,
                ])),
                Felt::new(BigInteger256([
                    0xdee382406a84c790,
                    0x968e213085e97764,
                    0xba6e0661f9d89ed3,
                    0x245fc019dbfa38b6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x95e3d87d8f3118b2,
                    0xda93f2ee0a1379aa,
                    0x9f027b3f9dcd90ef,
                    0x1ba2ba970832cd22,
                ])),
                Felt::new(BigInteger256([
                    0xe7104b761e379275,
                    0xf531cebb41eac083,
                    0x308fa75777b6c93f,
                    0x07127b5197a24282,
                ])),
                Felt::new(BigInteger256([
                    0x2da6e3795abb481a,
                    0x3334004faddc3a05,
                    0x6af9f90abbef0953,
                    0x1a736d1d8a94774f,
                ])),
                Felt::new(BigInteger256([
                    0xc467d64319039234,
                    0xff4c07540dd00755,
                    0x0b6ff914f3f23ff5,
                    0x15d9642e39eb8fa3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc500568b3d543a7b,
                    0xbc9df45fac661845,
                    0xa017c7493d0a65d6,
                    0x20a3979662522487,
                ])),
                Felt::new(BigInteger256([
                    0xa13cc388577b081e,
                    0xabd023c87a70eff3,
                    0xf1ca6cd1b23ce9f7,
                    0x2c6054890c9e9695,
                ])),
                Felt::new(BigInteger256([
                    0x3fb3dae0efbea1c3,
                    0x135afd757f595606,
                    0x31c6610bb8aa0dfc,
                    0x29ceb04ce949e403,
                ])),
                Felt::new(BigInteger256([
                    0x353767445b121d9d,
                    0x5a555e993a0febcb,
                    0xb8f1705d5a162c01,
                    0x0ccb25b9d495109f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x336459149db5b0f4,
                    0xb39ff09207a70e09,
                    0xd56bfea78ffc98bc,
                    0x0f50a0aa7d971197,
                ])),
                Felt::new(BigInteger256([
                    0x29da86cb85fa5692,
                    0x2c5ece79cbe3aed6,
                    0xac13196ee234c002,
                    0x01da8b59afba4cf9,
                ])),
                Felt::new(BigInteger256([
                    0x32ad9974f3b37084,
                    0x197f9ed3b1c9b73b,
                    0xd70e2f292a35d9e7,
                    0x15c5c232a80369a9,
                ])),
                Felt::new(BigInteger256([
                    0x5e4c901e9bf3cf3e,
                    0xacb97fe13ee67a60,
                    0xdc7066ac0b62df61,
                    0x0b3cb993fef9a67b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x725f3b2d251d547a,
                    0xf434c34732e486e1,
                    0xb2a954441c0950fb,
                    0x1cb58e8bec5c32b2,
                ])),
                Felt::new(BigInteger256([
                    0x8103ce56fd46affa,
                    0x5aeecab7a709b368,
                    0x4db0caec842a5536,
                    0x0d1106c1385c2119,
                ])),
                Felt::new(BigInteger256([
                    0xbc9df7794b2c06ab,
                    0xb6e26dba02fde376,
                    0x579b18174d6aeb3f,
                    0x2cffdda47fb3353a,
                ])),
                Felt::new(BigInteger256([
                    0xb1ade662dd4479b3,
                    0x71b53e268a6ac76e,
                    0xc8661c56665d5c9e,
                    0x1cb0b4b00bc12163,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4b4720d55ffa895f,
                    0x0f519f939a5fa692,
                    0x08db43423eba5a97,
                    0x0fe645d4e127e83d,
                ])),
                Felt::new(BigInteger256([
                    0x7a737905085e0c56,
                    0x4b2b1d7aa3070a58,
                    0xfb79fb81342931d1,
                    0x09e8edf7141b5643,
                ])),
                Felt::new(BigInteger256([
                    0x1972f846a23223a6,
                    0x5d99af87bd7080e0,
                    0x6511e8ffe2af2e72,
                    0x0c60ffb4c0594e98,
                ])),
                Felt::new(BigInteger256([
                    0x7b6f9c7f196f9251,
                    0xa90c21391e7dba88,
                    0xe814cc7f78a70140,
                    0x2b91460e3245ea76,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x75fbd43b813b5299,
                    0x5c9304e667ee1d42,
                    0x482bdf5b9090a2fc,
                    0x18298fe1c9660731,
                ])),
                Felt::new(BigInteger256([
                    0x41ac60dde1e28be4,
                    0x75f5385bcb807dd0,
                    0x75332824b2857fc4,
                    0x0f22df9e9a0544bf,
                ])),
                Felt::new(BigInteger256([
                    0xc7d9d96693a4c14c,
                    0x1104e7be5a0d55e8,
                    0x7e5b9a1e13ecc6ba,
                    0x04c7327ffb1ceffa,
                ])),
                Felt::new(BigInteger256([
                    0x9035159b41c59342,
                    0xa2f6bb397f9cf552,
                    0x3d3d8744ab9a3493,
                    0x111826a3cc7fa242,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd63d0ab5beb6b067,
                    0xe0bb7041d9991340,
                    0xd1c648d6f83c8258,
                    0x15a9a4a99c501d7c,
                ])),
                Felt::new(BigInteger256([
                    0x9252b399073e6b06,
                    0x282d50c01980d3f8,
                    0x936db7366a80dd67,
                    0x0233cb84a8bd3904,
                ])),
                Felt::new(BigInteger256([
                    0x27c2d3107696a55d,
                    0xb2a39a9ffefa3c8c,
                    0xcf50dcb2d4f47d47,
                    0x2ff37aa00d8f0a35,
                ])),
                Felt::new(BigInteger256([
                    0xdd1c7cee94c0c031,
                    0x5f4aef65ffec3693,
                    0xdba221fbc0535c5b,
                    0x24bc1267e7827dbc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf9d335a08df021ee,
                    0xa05bca1c6738b34b,
                    0xbca66eab48625e2a,
                    0x0369f3c17b729b38,
                ])),
                Felt::new(BigInteger256([
                    0xcdbab59e0b1dbdd3,
                    0x8d472c4ef1ddabfe,
                    0xb281216db6b96b15,
                    0x1bc958943253c902,
                ])),
                Felt::new(BigInteger256([
                    0x59bfe2fb73b0a661,
                    0x1559ebfb482605a5,
                    0x8f72f93ee879aa89,
                    0x079ca365d708aa9d,
                ])),
                Felt::new(BigInteger256([
                    0x6db1a77e6c8919a1,
                    0x0543b11ecda2c5eb,
                    0x0a0f122c695c6d7e,
                    0x0a8a17facd13275c,
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
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
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
                    0x3a6c045f23913279,
                    0xb2fa39a1556ea376,
                    0xa96fc9c63b64be3f,
                    0x2f21b9c8d77021cd,
                ])),
                Felt::new(BigInteger256([
                    0x633de0e69a4d6b3f,
                    0x4621f288e4a752df,
                    0x0bf89582f2b74d46,
                    0x1b07a551e1688fc6,
                ])),
                Felt::new(BigInteger256([
                    0xcf07bffe01ac78e3,
                    0x34eaa2b45b178c7e,
                    0xde0bae25accc1a84,
                    0x034408f2b4fb8014,
                ])),
                Felt::new(BigInteger256([
                    0xeabf532fc39498b7,
                    0xf6cf0caf965d6ffa,
                    0x14f50d3f8d5a554d,
                    0x0bfdbd60bf32b3b2,
                ])),
                Felt::new(BigInteger256([
                    0x03a3923aaccf9220,
                    0xeabe363df3290954,
                    0x6270b666935d6375,
                    0x191d593832af408f,
                ])),
                Felt::new(BigInteger256([
                    0xeb4ee46fa440c5f5,
                    0xe24a2e38a7b0a774,
                    0x7080640b3a721a8c,
                    0x04054f37d5640b0b,
                ])),
                Felt::new(BigInteger256([
                    0x42aeb875ac6113fa,
                    0x1be0d0a9eeee278c,
                    0xf552721d3700294c,
                    0x18736b8b66bd1891,
                ])),
                Felt::new(BigInteger256([
                    0xcd0ecc71f29f8dd8,
                    0x7b0f0d49d5ab6d6e,
                    0x9b729d3d7b2cda3e,
                    0x0d1e068581eba5b5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x715802301a8728af,
                    0x5c7e17c156b388dd,
                    0x267e41e3f790aa11,
                    0x2345002fbad90937,
                ])),
                Felt::new(BigInteger256([
                    0x8ef1f6ea49c5af22,
                    0x6bf7219a508eb832,
                    0x5cbe6d47c9b5d0fb,
                    0x0b461afac1578a5f,
                ])),
                Felt::new(BigInteger256([
                    0x863fff9a8e2c7b47,
                    0x11bd52d9cdc5dfc2,
                    0x4a5b8ca16f2230fd,
                    0x210a2b19638bcd3e,
                ])),
                Felt::new(BigInteger256([
                    0x00a024d36e379f03,
                    0x280016af71a6e1b3,
                    0xe50b44bbf09a145b,
                    0x28469d350621e4c3,
                ])),
                Felt::new(BigInteger256([
                    0x182fa6df013074c8,
                    0x95d080be54333307,
                    0x5b9e916b6a2ad76f,
                    0x225df88221cf6e8d,
                ])),
                Felt::new(BigInteger256([
                    0xfb0611cc158d7b93,
                    0x8bdf890796020d89,
                    0x4c14555f84109551,
                    0x1c1d4cf0ad7bf4df,
                ])),
                Felt::new(BigInteger256([
                    0x9cceeed4ba3abe21,
                    0x2e1e1882c6ae55ee,
                    0x7fe4991975466afe,
                    0x0db9627de5ebaedd,
                ])),
                Felt::new(BigInteger256([
                    0xfc2be2aa9846cb52,
                    0x73699d3db08fc6ea,
                    0x4dca771d7a996a7b,
                    0x0e617596696b32a2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa4e7cbed676d4ecf,
                    0x294f49a2ec673658,
                    0x48cca71cb7997b79,
                    0x0c74fedd14bfbbeb,
                ])),
                Felt::new(BigInteger256([
                    0x8412e2a31486b34f,
                    0x9de660f8a5dbf7d1,
                    0xf3b781a4c0dee407,
                    0x1f74f8da43f30e24,
                ])),
                Felt::new(BigInteger256([
                    0xc4bdb8a7aecc2379,
                    0x353ab22036e2de0b,
                    0x1ed4ba59786db0dc,
                    0x1a33b174d52f81ea,
                ])),
                Felt::new(BigInteger256([
                    0x62bc5dd7d9b1dce1,
                    0x59ffa81c99a501a3,
                    0x8a3add0c95a2422c,
                    0x151d2fdb2d00c908,
                ])),
                Felt::new(BigInteger256([
                    0x1d6f8a2d0a0dca8b,
                    0xaa974ffea9795c6c,
                    0x99ae532597c09839,
                    0x17cceaa62b07d427,
                ])),
                Felt::new(BigInteger256([
                    0x3d09d58dcf13ef30,
                    0x59e47a719e10593a,
                    0xc951dd28364a1717,
                    0x24d39260e48185e5,
                ])),
                Felt::new(BigInteger256([
                    0x45676c0b799339d4,
                    0xa605be37d37394fc,
                    0x210e0029d130dc8b,
                    0x2497af81db9ffbb6,
                ])),
                Felt::new(BigInteger256([
                    0xf694f4c835fe6028,
                    0xd30db2f4181f1dc2,
                    0x6e37ec963f913055,
                    0x2d75164e0a28eea4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xff865df08d73e1c8,
                    0xf383e70d6ca3a5f8,
                    0x0a771552dc7edad6,
                    0x2f361ea99bdda89a,
                ])),
                Felt::new(BigInteger256([
                    0xb45505cae6bc39e1,
                    0x2a120e7d108ecc87,
                    0x5b33c311ce10d58a,
                    0x1cb32dd565aa7932,
                ])),
                Felt::new(BigInteger256([
                    0x30eec692451a5821,
                    0xa08c85ee32c4efce,
                    0xd243e326c6ce8adf,
                    0x009429a9fe7f34f0,
                ])),
                Felt::new(BigInteger256([
                    0xd70f435dc524097f,
                    0xcf2a14b3c66dda8d,
                    0x239f94b448f0f2b9,
                    0x27219540e8cad47d,
                ])),
                Felt::new(BigInteger256([
                    0x698816a1b8b71048,
                    0x0d00452097971607,
                    0x5dc9236b5a0f7b3d,
                    0x125477b6ac2c8255,
                ])),
                Felt::new(BigInteger256([
                    0xb7e625e8c6088a7d,
                    0xfdee3119673e6441,
                    0xbb0ce1deac334a0d,
                    0x2f80e5a5b15446c5,
                ])),
                Felt::new(BigInteger256([
                    0x909b9435d8ca2429,
                    0x8a3799b4f6ef88b2,
                    0x68f66fbad4f25eaa,
                    0x2f69e50a0ba92d4f,
                ])),
                Felt::new(BigInteger256([
                    0x05cabe0ce11ea4e3,
                    0xa0afe047f72457f9,
                    0xe15fd070f333629e,
                    0x0ffc0ba0df65f461,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc7c427a1ac1803f8,
                    0x31d9adbc6a66ae14,
                    0xd5340d70357bcb9b,
                    0x1f0d44f2ebaef984,
                ])),
                Felt::new(BigInteger256([
                    0xb993b98e5059da17,
                    0x61cffd9f1be7dd6d,
                    0x822bfd91b72288d3,
                    0x2784e5060b235362,
                ])),
                Felt::new(BigInteger256([
                    0x58d1cb6379263866,
                    0x1d80664d77d4a076,
                    0x26c409ac0fcf2009,
                    0x03d41289bdcece66,
                ])),
                Felt::new(BigInteger256([
                    0x5fdbbac32283b805,
                    0xe6b2ba8226dc87d3,
                    0x34bd7fd0bd6328bd,
                    0x08653b5ee3d44e9f,
                ])),
                Felt::new(BigInteger256([
                    0x2b5bebd1f3cb26a7,
                    0xdaeea2f4684b4e37,
                    0x93adfcf640f21d3d,
                    0x1ff64c8819e5092b,
                ])),
                Felt::new(BigInteger256([
                    0xdee2970185d0185c,
                    0xa108fe9ceac99747,
                    0x462e169ff91afa42,
                    0x01988baa7573b4e2,
                ])),
                Felt::new(BigInteger256([
                    0x8ec24d29ea5c7b40,
                    0x24f1768735cbb6ad,
                    0xa5bb9a64ca0fbf24,
                    0x051242cfe8d05583,
                ])),
                Felt::new(BigInteger256([
                    0x43e0e744c64af24c,
                    0x4faa128e9b30464f,
                    0xab8f28deec18ae24,
                    0x00233d8f8a886372,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xac2dc4224c688d00,
                    0x24832b99bda2449d,
                    0x01d4dc9c401b7509,
                    0x12ff5ec42dd03d19,
                ])),
                Felt::new(BigInteger256([
                    0x0ee4c4de3ce74682,
                    0x4c1271fe506f9218,
                    0x62853b126f112e32,
                    0x12d92203eb0fdf1b,
                ])),
                Felt::new(BigInteger256([
                    0xaf731125a0454f3e,
                    0xb1aa47cd773555ae,
                    0xfa0065df3e19a766,
                    0x1cad14b024826a1b,
                ])),
                Felt::new(BigInteger256([
                    0x8b2fd75fa08df536,
                    0xb4c97df8302cee02,
                    0x568f503b25a161ba,
                    0x3052f2f3376c6132,
                ])),
                Felt::new(BigInteger256([
                    0x58dc755b0d2dc880,
                    0x4720ca5eaa3e4dd5,
                    0x7cb4386f4641797d,
                    0x07003eb295976784,
                ])),
                Felt::new(BigInteger256([
                    0xb7be468a3031dfb4,
                    0xce69fe029c85e7fc,
                    0x7f6bfda1e0f107dd,
                    0x1aa8e00c0b8b65b5,
                ])),
                Felt::new(BigInteger256([
                    0x226376dc969c2838,
                    0x94ae0a148fbb0132,
                    0xb88c2bd5bcfe9f75,
                    0x1a036d986a03aaaf,
                ])),
                Felt::new(BigInteger256([
                    0xadda5f75c76736b0,
                    0x6df5e2b13a9188ad,
                    0x4d4e747d95240274,
                    0x10f64fc16ad23b00,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xde425301065b6fb4,
                    0xe683adb174e07439,
                    0x7734e76720c53324,
                    0x2b233dc4ea4b12c7,
                ])),
                Felt::new(BigInteger256([
                    0xfddd40ed76b444ff,
                    0x3ec4aff022a04848,
                    0x9b70e9b5a98636af,
                    0x0ed8ea17616407b4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xed16384d3da9769a,
                    0xcb46ab0ed1957e2d,
                    0xf6b610b583e85404,
                    0x14cbb363d850444a,
                ])),
                Felt::new(BigInteger256([
                    0xca272e4554a8f1cc,
                    0x9cde6d612bfa60ce,
                    0x4532301a806d7eae,
                    0x018673e2a9c8d1ea,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x876a2fe0116f6385,
                    0x764688ac4f7de922,
                    0x51ac2e93d83b41e5,
                    0x05b1d941b195a448,
                ])),
                Felt::new(BigInteger256([
                    0xab7821b9373b24a9,
                    0xf47dd60f4fbac7d9,
                    0x3bffa06c6ba90935,
                    0x1cebdf7fd18dd225,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc893a5555495def7,
                    0x38778743c34da3be,
                    0x198de29e74331b75,
                    0x1a0df9706a6a6861,
                ])),
                Felt::new(BigInteger256([
                    0x9a539eb5da102874,
                    0x6ea417d04c0f1131,
                    0xf26b97788ad1bd9b,
                    0x08c72bd00002070b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2c3ebe9c6524a14f,
                    0xa8487ac0e543e13a,
                    0x86037391ce5f5dee,
                    0x0055350d9302a712,
                ])),
                Felt::new(BigInteger256([
                    0x5c207e9911aac0ae,
                    0xee2b5de7450d3da8,
                    0x7866d893b3ea60a7,
                    0x1efda2647ee17639,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1ae5c221114b1886,
                    0x115f719710e544ae,
                    0x16c52076f29ea3fb,
                    0x059285ceeaa0c5bc,
                ])),
                Felt::new(BigInteger256([
                    0x1fa0ba3f488eb4b1,
                    0xeb588a68f90977cd,
                    0x50d042052e165f68,
                    0x17b39240b0531588,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb90ce0876a1cb918,
                    0x48ac1a40b89e2d10,
                    0x87c798d25faf4bc6,
                    0x161a019247b6d054,
                ])),
                Felt::new(BigInteger256([
                    0x725f4abe9a853c04,
                    0x498090ccca3917f0,
                    0x5b758aaf877b6fc8,
                    0x113bbc1c35a008d9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa8b9f39341dca0a0,
                    0xebabf71ff12edc9d,
                    0x445abc01badce434,
                    0x149a7735db2fefb4,
                ])),
                Felt::new(BigInteger256([
                    0xc3dcbc5fc8a50a44,
                    0x3860a3d9e4d1620d,
                    0x7399a7896c1587cb,
                    0x1dc7d2b2f3987c1d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x33f61edcb2ef28da,
                    0xa3511708380d9210,
                    0x1352c43ea03ea2fb,
                    0x006f07628c8162b8,
                ])),
                Felt::new(BigInteger256([
                    0x6d44445eb2b0dabf,
                    0x2e4d4aab6ec51415,
                    0x09d5c39c5833809f,
                    0x06fa87ccc4783300,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9578448ebab18f99,
                    0xb9cb9db640edcc5b,
                    0x0b92b750797df609,
                    0x220c3c11536fb593,
                ])),
                Felt::new(BigInteger256([
                    0x974cb33ba2c31ee2,
                    0xb3f12175991b4335,
                    0xaef1a7579b179736,
                    0x19c12caebe7f99e5,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
