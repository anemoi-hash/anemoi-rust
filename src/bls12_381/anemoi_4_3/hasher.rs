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
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger384([
                0xe301ffcdbc8c37c6,
                0xb794a919e29881d5,
                0x4c9579f4e6a4fe0f,
                0x770d7dd646cb10f4,
                0x683e66d49d523dd9,
                0x12dfa1d548d64769,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x53081d2afefa5a6e,
                    0xd4fb5b81d9f7cbd2,
                    0x6ddd20bb4c4b5893,
                    0x2df61a9fecad625b,
                    0xf135f87c24ec8904,
                    0x113726ecf87aaa69,
                ])),
                Felt::new(BigInteger384([
                    0xc410df14ffb230e3,
                    0xf2bdda7de5aa7775,
                    0x1809f9c40040af7e,
                    0x412a9ba68f0236ec,
                    0x8c0723eb4a75a907,
                    0x05a2c11204d70664,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x732ddc04279cd217,
                    0x25c4e86a4c4e7932,
                    0x10174c011706598c,
                    0x1ec205e27ba1fd39,
                    0x92303d38c5906643,
                    0x1333d41a29294bd3,
                ])),
                Felt::new(BigInteger384([
                    0x30b7a5ca17232960,
                    0x74bfe281d0c4323c,
                    0x12cd21065f004703,
                    0x1dfd44ae937efe3e,
                    0x426aba803724fd3e,
                    0x128e87ae814dc59d,
                ])),
                Felt::new(BigInteger384([
                    0x50c5d8929ea818d3,
                    0x08f885cb49d50f3c,
                    0x8de04fd02caa808d,
                    0xdb76d93487151d97,
                    0x9066d83358331459,
                    0x02a280a60d1dcf74,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xdf7c40e1ce2897e4,
                    0xa201c1422151da49,
                    0x292420c2b7a6cb5f,
                    0xbf7162c7a68d7156,
                    0xe3d64fcd320b3322,
                    0x12dc723f6d0f493b,
                ])),
                Felt::new(BigInteger384([
                    0x598c32ced81d884c,
                    0xce3a1f063006a74f,
                    0xaf6fe566340ad40e,
                    0xacd867669515e1d3,
                    0x078ded89b693e252,
                    0x096b36cbb4cf0419,
                ])),
                Felt::new(BigInteger384([
                    0x5b43b8d93803fb01,
                    0xfbaeda18d217e422,
                    0x18796b9ed6276ecd,
                    0xa9a8e5471ade4621,
                    0x5db937becd8696d4,
                    0x0fac5ba6d69be9d0,
                ])),
                Felt::new(BigInteger384([
                    0x426a3a55ecc22df7,
                    0x50fa44203ec329e1,
                    0xb0c7d984aec2545f,
                    0x4da5431afddb5ea8,
                    0x97c2eed2f3f00f26,
                    0x022e1cc413db3661,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x1ede80c645125e9e,
                    0xd7c594c5f300a9af,
                    0xf1946f4e408b61e5,
                    0xdca599ed289b9749,
                    0x86ebcdd7eb83072e,
                    0x03b64b9c43239ad7,
                ])),
                Felt::new(BigInteger384([
                    0x5c098c79c6935e03,
                    0xad8aa446c3d69dd5,
                    0xc90616dfaef6ba00,
                    0x6fd6e0b0ac6b2fe2,
                    0xc7d467b08025d35e,
                    0x16bd7b2ff9205084,
                ])),
                Felt::new(BigInteger384([
                    0x8d6986694034fdd6,
                    0x2da60d63c9079f69,
                    0x2c12a849c57d09c5,
                    0xe590db22b9f4f508,
                    0x032de440caaafcc8,
                    0x1530545ad171a847,
                ])),
                Felt::new(BigInteger384([
                    0xf5946dedfaf93a90,
                    0x28c6f70fe2940816,
                    0x478407b29e646ad2,
                    0xd55ee000e57f5603,
                    0x667c3d8c25b70e81,
                    0x12a95d28e99d8899,
                ])),
                Felt::new(BigInteger384([
                    0xe3bd93aec060a2c5,
                    0x6b0ffcc4728bf6e5,
                    0x2b114d9b58dee503,
                    0x433c5ebf23baf892,
                    0x8955f6cddc793614,
                    0x11e277153d4a6a69,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xec1e5f359c9b791d,
                    0x6677a03632a61c0f,
                    0x736ca1580121b243,
                    0x1493f30ae9ceca1f,
                    0xf52a22aca7a85a13,
                    0x04aacc03d8bb0b0d,
                ])),
                Felt::new(BigInteger384([
                    0xc2a9d0d3415a2b71,
                    0x9854e1d357ae49dd,
                    0xcd05b5b0fffdb33f,
                    0x5dfa69d336d7e344,
                    0x48cb33b1562d8969,
                    0x1399b70fe97d25d5,
                ])),
                Felt::new(BigInteger384([
                    0xb1422220e7f27d36,
                    0x8faa8cdd49d8ecbe,
                    0xeb7bf3aa36985b20,
                    0x299e619ce5c0943b,
                    0x2d22efd34d3809d9,
                    0x10254a586edab8a6,
                ])),
                Felt::new(BigInteger384([
                    0x79bb6f6e3d040f6f,
                    0xe513b16a356f2fd7,
                    0xbe6b08fd9332eacc,
                    0xb9fbfbb9cfbf5671,
                    0xf12e8dc9e16d4e82,
                    0x05cd659e00afd2ff,
                ])),
                Felt::new(BigInteger384([
                    0x950b52da6fac3f6d,
                    0x7d91545cf2878886,
                    0x0f6991cef6a32892,
                    0xca628a26e08b7472,
                    0xd8b2055e748b257e,
                    0x152e744abf35ea3c,
                ])),
                Felt::new(BigInteger384([
                    0x49f723fb9f025a0f,
                    0x87693df9ba70c08b,
                    0x13686a5fa038d9a6,
                    0x21bc7088fdd850b2,
                    0xc1e23ca8170e83d6,
                    0x16b8fd9198ec6cff,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x39fbd5200c2bac10,
                0x95f81e0e1307e387,
                0x397ce456a94c4947,
                0x123cf4c1011d8bf3,
                0x71d9524c7bf6769d,
                0x09ad95eba7f95572,
            ]))],
            [Felt::new(BigInteger384([
                0x994252f22dfd3486,
                0x10444e4eebed5790,
                0xfd915619d8ccb608,
                0x4cace64880248a62,
                0x441bd56aab94d0bd,
                0x19109469c94da4ab,
            ]))],
            [Felt::new(BigInteger384([
                0x3ecfc09d7acc56a6,
                0x072139b18f16f065,
                0x9f9e6f9b6a92d20b,
                0xf05b0bdce8cbac08,
                0x4dfcccee2a8e468e,
                0x1084add38f374014,
            ]))],
            [Felt::new(BigInteger384([
                0x72e3b70c36c9c6ae,
                0x8b8af13d258dd0ac,
                0x4168ff6869fcafec,
                0x631f7259f4297b22,
                0x056d21a5f810ffe6,
                0x086ec726a2b276ea,
            ]))],
            [Felt::new(BigInteger384([
                0x907923f9ee7dc527,
                0x45f8eaccc49d1f11,
                0x8f55481a6e365eb2,
                0x2339286c8b6be083,
                0xaa647aed8aef9acf,
                0x0509201a7c824665,
            ]))],
            [Felt::new(BigInteger384([
                0xbcf165bf9f47dd51,
                0x5a35b153f36af977,
                0x7cd68e45aa452467,
                0xff0946dd1fba6be7,
                0x3ed4d1cd9842395e,
                0x18b84d9e18c98e2a,
            ]))],
            [Felt::new(BigInteger384([
                0xfdb631dee5877662,
                0x09ae255d7bf354d7,
                0x1bf30b89ce2060d9,
                0x00185451286e25e7,
                0x37d12887d8c8c5f2,
                0x096e97e0d13a08c3,
            ]))],
            [Felt::new(BigInteger384([
                0x6f3c55efece55b21,
                0xb6c1f7a32f8e13f5,
                0xd755a785f0a5555b,
                0x31a5b544d5093f0a,
                0x014c6f36b89bafc8,
                0x06efd91dc209cdb4,
            ]))],
            [Felt::new(BigInteger384([
                0x380baa2f4342a283,
                0x3202be02cf5c8229,
                0x1c836518d89857f1,
                0x03d7697e057ad046,
                0x223466155c6ff86f,
                0x113dd7d12c55e351,
            ]))],
            [Felt::new(BigInteger384([
                0xd1b8ffc8538767ab,
                0x833381a4945f525e,
                0xa57291631871750e,
                0xf2196f80aa761d70,
                0x9b8ca76ede3caf2b,
                0x0341b24f74136798,
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
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x39fbd5200c2bac10,
                0x95f81e0e1307e387,
                0x397ce456a94c4947,
                0x123cf4c1011d8bf3,
                0x71d9524c7bf6769d,
                0x09ad95eba7f95572,
            ]))],
            [Felt::new(BigInteger384([
                0x994252f22dfd3486,
                0x10444e4eebed5790,
                0xfd915619d8ccb608,
                0x4cace64880248a62,
                0x441bd56aab94d0bd,
                0x19109469c94da4ab,
            ]))],
            [Felt::new(BigInteger384([
                0x3ecfc09d7acc56a6,
                0x072139b18f16f065,
                0x9f9e6f9b6a92d20b,
                0xf05b0bdce8cbac08,
                0x4dfcccee2a8e468e,
                0x1084add38f374014,
            ]))],
            [Felt::new(BigInteger384([
                0x72e3b70c36c9c6ae,
                0x8b8af13d258dd0ac,
                0x4168ff6869fcafec,
                0x631f7259f4297b22,
                0x056d21a5f810ffe6,
                0x086ec726a2b276ea,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 188];
            bytes[0..47].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..47]);
            bytes[47..94].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..47]);
            bytes[94..141].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..47]);
            bytes[141..188].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..47]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xd26588031af165c1,
                    0x88f666294d779929,
                    0xf8bdf9bf6fe7fe74,
                    0x2abb610a38195d58,
                    0xfbdd69b049326e35,
                    0x0ccc6ac6f7698fe9,
                ])),
                Felt::new(BigInteger384([
                    0x4e8e8ffe4eeba6c8,
                    0xcc9d84b8a31c0343,
                    0x1e233e888ac86df1,
                    0x8450394676d93cf0,
                    0x7f14e0ce81c44461,
                    0x0505ded5abb6504a,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x8ca275b6546a1b1c,
                    0x385061693455d865,
                    0x3895e15b712e887a,
                    0x4c8a3988438e5d64,
                    0x8fd35f970f7a9ec2,
                    0x18da154eccc4f838,
                ])),
                Felt::new(BigInteger384([
                    0x7e73b472c9c4797f,
                    0xb338363d555688f7,
                    0x22bd0c1e30b28abc,
                    0x5234fc4783edfa0e,
                    0xdb7bde75f67410c5,
                    0x15c8d1b2b0dd52b6,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x977ee0f0abddf716,
                    0x7f62a4b3d2e7992b,
                    0xa4f6369a75e45162,
                    0x01212b75e207e520,
                    0x84ce5b00b58bcae2,
                    0x0d3b4e9aa1494d32,
                ])),
                Felt::new(BigInteger384([
                    0xf128f2dd2de977ad,
                    0x5cbbad39b4cac532,
                    0x3aa27c07ed4c25d7,
                    0x121c35fd0f032b41,
                    0xa49ca3597e8c9b87,
                    0x0ba55284725394ba,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x40d44e92e7b00295,
                    0xc6ddb0a36a01df2f,
                    0xba85a6ddd9fe5840,
                    0x059fb1431251e86e,
                    0x3b75e16cea3736ef,
                    0x14fd7e52fea7e0d8,
                ])),
                Felt::new(BigInteger384([
                    0x7688e82fe19b3620,
                    0x4f0b46c13ba88a15,
                    0xf91d920c7abedc8c,
                    0x7bd0e271b726fee5,
                    0xcc162c16f9365359,
                    0x07069f27cebbca59,
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
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x20f4180169dd0c89,
                0x5593eae1f0939c6d,
                0x16e13847fab06c66,
                0xaf0b9a50aef29a49,
                0x7af24a7ecaf6b296,
                0x11d2499ca31fe034,
            ]))],
            [Felt::new(BigInteger384([
                0x51172a291e2ee9f0,
                0xccdc97a7d858615d,
                0xf4221ad8ab301d12,
                0x3a47ea4ad3f744b2,
                0x20339656c2a302b0,
                0x14a1d51744226455,
            ]))],
            [Felt::new(BigInteger384([
                0x88a7d3cdd9c76ec3,
                0xdc1e51ed87b25e5e,
                0xdf98b2a263307739,
                0x133d6172f10b1061,
                0x296afe5a34186669,
                0x18e0a11f139ce1ed,
            ]))],
            [Felt::new(BigInteger384([
                0xfd5e36c2c94b8e0a,
                0xf73cf765f4566944,
                0x4c7266495e0c3ea8,
                0x1cf9482fd5f3d495,
                0xbc7065cda021dd71,
                0x02030b9093e3c497,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
