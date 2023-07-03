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
                0x5a4bdf66e80fa401,
                0xdb5cfe4169f8b63c,
                0x5d9dca0e9c16a71a,
                0x0f4f5c0063990f0a,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xcaf9996b6f0c2d73,
                    0x45163a9ae60b2ea8,
                    0x5350e09a4ef3f03d,
                    0x0689146b51c3cf14,
                ])),
                Felt::new(BigInteger256([
                    0xf599cbf369d2d5b0,
                    0x6700e98ffb736831,
                    0xb5cb7645538cc94c,
                    0x016b39e1454d0f8d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xce3a2139fcc523de,
                    0xee9921f60d2670ca,
                    0xca8bd04912d98940,
                    0x112a7a82bc882fb0,
                ])),
                Felt::new(BigInteger256([
                    0xea562aee96ebf730,
                    0xe664b1c40cf61fc5,
                    0x89117d8ba2e3ee84,
                    0x079509649a95acb6,
                ])),
                Felt::new(BigInteger256([
                    0x07eabbf3f9787f6c,
                    0xb499a8048c1e3db4,
                    0x6b0ad98c6df074ae,
                    0x0cec10854a3ce3aa,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc8756615aea48655,
                    0xa1c4a7e03eb3373c,
                    0xcd2b539265ff22ba,
                    0x106f83c25ab2c386,
                ])),
                Felt::new(BigInteger256([
                    0x9746413942cfcb5a,
                    0x32092cec7efdee3b,
                    0x7daadfce8b240ba0,
                    0x108c125057cd0376,
                ])),
                Felt::new(BigInteger256([
                    0xf6bc122dd66fe9a9,
                    0xb91605a7d701015e,
                    0xfc9818d4f3945724,
                    0x11b158a418e0ece1,
                ])),
                Felt::new(BigInteger256([
                    0x4cbfbd7e55fd7d78,
                    0xeb8a7c1cd614869d,
                    0xad47b8be7c5732f5,
                    0x058d0020d27e3060,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x806275144dd6da00,
                    0x58ed7fb78c8f7eb6,
                    0xbc3747b7844d084a,
                    0x0a79595cf62210db,
                ])),
                Felt::new(BigInteger256([
                    0x101669ebf9b31024,
                    0x98d501f9630e5051,
                    0x2e189d2004fe36f6,
                    0x027edb7aa50141f8,
                ])),
                Felt::new(BigInteger256([
                    0x748c43611a07932c,
                    0x56fec25f5c4a7f57,
                    0x18696bea96e46b3a,
                    0x0501d0c2f19d9fff,
                ])),
                Felt::new(BigInteger256([
                    0xa8aecde5073010a2,
                    0x82668944e3b5fb53,
                    0x2c08f4a3d8661537,
                    0x02d77076bd5c3658,
                ])),
                Felt::new(BigInteger256([
                    0xab8644866db45c91,
                    0x5759a9a2b776b9b4,
                    0xbd3369ebf68fd158,
                    0x032e071ed6bc3fa2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9223b17ffb34b889,
                    0xb8055cd0803cb032,
                    0x3c06ea3a24435cf4,
                    0x088892014f6fdeb3,
                ])),
                Felt::new(BigInteger256([
                    0x1d2cb0a1ff66ac22,
                    0x464b23cb9f7b6050,
                    0x99274bf644deccd1,
                    0x05fb6d8e00dcc0cd,
                ])),
                Felt::new(BigInteger256([
                    0x95c2d3c68d3e5b19,
                    0xeb688ffab4c81cef,
                    0x5361019834d108ba,
                    0x00f4c74957c9af56,
                ])),
                Felt::new(BigInteger256([
                    0xcc2ceebb25658245,
                    0x44bf016d07722dd0,
                    0xf672a84fbcceac23,
                    0x07c2160270966d65,
                ])),
                Felt::new(BigInteger256([
                    0x5ee07420aebe1b9f,
                    0x3514f4f9656c8f51,
                    0x85d981e58f73b4cc,
                    0x0ecd8394f40d11d6,
                ])),
                Felt::new(BigInteger256([
                    0x372cff60466f5500,
                    0xcaa7ad28c3f1503a,
                    0xefdb9b7df614afb6,
                    0x0c4db54dfc910ee8,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xaa9afdc7dbd0f9de,
                0x7b64cd1d7991121d,
                0x11975b6566b8a4ed,
                0x11a8b4d198c6fd0f,
            ]))],
            [Felt::new(BigInteger256([
                0xc7df5db301209d05,
                0x534f2b782598ca70,
                0xff9d34f88c98f5d3,
                0x01ae96a835b35d7c,
            ]))],
            [Felt::new(BigInteger256([
                0xd7d9a76d248cb96f,
                0xeeec672219b3e770,
                0xe3d957138075c7d6,
                0x0ee3960ff6cb106a,
            ]))],
            [Felt::new(BigInteger256([
                0x95660ce14b05a3f4,
                0x95e759c58edd998c,
                0xdf49034c4d2c34b9,
                0x05e43f2b342ce0cd,
            ]))],
            [Felt::new(BigInteger256([
                0x4039b3065eec608a,
                0x590b738b909b1829,
                0x9e5a85c0179274f0,
                0x02b00971fc037ff8,
            ]))],
            [Felt::new(BigInteger256([
                0xab85d4e1aa7cb667,
                0x26555f58e74582ba,
                0x40276f516037bd51,
                0x0cd97d1b0646e96c,
            ]))],
            [Felt::new(BigInteger256([
                0x079f959b7c20d690,
                0x9faf3987f3c06236,
                0x1a1a8104f6ad0e29,
                0x035203ecf3fd3df6,
            ]))],
            [Felt::new(BigInteger256([
                0x1bb770aee2f0206b,
                0x27b421adeebf0185,
                0xeb6d13d47289d13c,
                0x025b01db90f57eb8,
            ]))],
            [Felt::new(BigInteger256([
                0x92d0f5f1caf1caca,
                0x837154137395797e,
                0x33b1314eb109a4b1,
                0x0ee7153e146e69d8,
            ]))],
            [Felt::new(BigInteger256([
                0x9177d73b10fc5342,
                0xe550135629db2370,
                0x34c1ee0b86b2fe32,
                0x03f40a4f4b1c7943,
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
                0xaa9afdc7dbd0f9de,
                0x7b64cd1d7991121d,
                0x11975b6566b8a4ed,
                0x11a8b4d198c6fd0f,
            ]))],
            [Felt::new(BigInteger256([
                0xc7df5db301209d05,
                0x534f2b782598ca70,
                0xff9d34f88c98f5d3,
                0x01ae96a835b35d7c,
            ]))],
            [Felt::new(BigInteger256([
                0xd7d9a76d248cb96f,
                0xeeec672219b3e770,
                0xe3d957138075c7d6,
                0x0ee3960ff6cb106a,
            ]))],
            [Felt::new(BigInteger256([
                0x95660ce14b05a3f4,
                0x95e759c58edd998c,
                0xdf49034c4d2c34b9,
                0x05e43f2b342ce0cd,
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
                0x509ad2afb325468c,
                0x1ddec192d5dc34e6,
                0x2cbae14c826cb34e,
                0x0ca2afee80c9f5ca,
            ]))],
            [Felt::new(BigInteger256([
                0x8a99f7075868f682,
                0x22013da2817e404e,
                0x1366adc934853063,
                0x112085779fcacc10,
            ]))],
            [Felt::new(BigInteger256([
                0x0835de201f261251,
                0xf1ed1c18431bb5e0,
                0xbc24640f57d9de13,
                0x03bb180d0cff7bb1,
            ]))],
            [Felt::new(BigInteger256([
                0x8db10d40be3e8ec8,
                0x479da8e5dfd3b544,
                0xd9d90ab51b5fd67b,
                0x02d3bf4e53830470,
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
