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
    use super::super::BigInteger256;
    use super::*;
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
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
                0xf864304efe377bd7,
                0xc9a9b5377a71cd16,
                0x1eab615077a87b6c,
                0x02e3288d4327897a,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x2f51799928c52117,
                    0x2c0938489f291a20,
                    0xb589f8669036f7a6,
                    0x0bd26f4ea7032d3c,
                ])),
                Felt::new(BigInteger256([
                    0x9d6b4893aff95c80,
                    0xe25cd371388d6c97,
                    0x760fa8b0f46a156f,
                    0x115c43b8beba0288,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x750bd379155798e9,
                    0xcbcb4c87bd598f6e,
                    0x92e75a892bb8bddd,
                    0x0e14d2b6c43d1048,
                ])),
                Felt::new(BigInteger256([
                    0x2d7e04bcbedb89e2,
                    0xdbe433659ed8257a,
                    0x70f41209a2fda430,
                    0x03c04f2a9bd53d11,
                ])),
                Felt::new(BigInteger256([
                    0x31f1319042190159,
                    0xc94cb5647eacc31b,
                    0x69706bbda2192089,
                    0x0a158911ca422a78,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc97e924f2863503a,
                    0xd6bd53e1b11c7a18,
                    0x39e48e622c1b84b0,
                    0x04b8d7005a193f7c,
                ])),
                Felt::new(BigInteger256([
                    0x8d4c18e9a56556f0,
                    0xade98d42ab12b041,
                    0x43f07d993e200f88,
                    0x0d139a1253b73844,
                ])),
                Felt::new(BigInteger256([
                    0x2af95334911a7d83,
                    0xc670d0228a754801,
                    0x2baf3ab7873db385,
                    0x02a49b1a996cdbf6,
                ])),
                Felt::new(BigInteger256([
                    0x3354c4fb529f403f,
                    0xbe57076095415c78,
                    0x0117b1f84fa9068c,
                    0x0300fe0f18c72f96,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xea81c11fac6d98a7,
                    0x51759dcd5172c60b,
                    0x8d26f83b75b709b3,
                    0x058f2305a7f64359,
                ])),
                Felt::new(BigInteger256([
                    0xac8498e2af5cf0b3,
                    0x855c5b94a38da74a,
                    0x6e45cb53fc583d01,
                    0x0e9affb645d21d93,
                ])),
                Felt::new(BigInteger256([
                    0xe9d2f87d1313bd88,
                    0xf4a410010e003422,
                    0xff173685253786bf,
                    0x0795ca9642ee730f,
                ])),
                Felt::new(BigInteger256([
                    0x157190de1c02364e,
                    0x08e0cd8e6dfbb650,
                    0x8a813609dac490fa,
                    0x02fc8176e37ae23c,
                ])),
                Felt::new(BigInteger256([
                    0x6d594c631f41aab8,
                    0xdb3cd652d3f289ca,
                    0xcb58915c84c33f3f,
                    0x01a7c5d55e9f9161,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5592937bbcac618e,
                    0xcddb65393f28a2ea,
                    0x67eb481d2cf67331,
                    0x0439cbfb0a72ab93,
                ])),
                Felt::new(BigInteger256([
                    0x32a9dc296a732c39,
                    0x33c3410fc812ac02,
                    0x23fa55e0e90e15c1,
                    0x1201d18c90be6881,
                ])),
                Felt::new(BigInteger256([
                    0xdc7b1f56dc816aa1,
                    0xc2916bbaaf82276d,
                    0x5efce09db2003c49,
                    0x01bc9c8320fa53ea,
                ])),
                Felt::new(BigInteger256([
                    0x98fad5e46067c8bb,
                    0xe1c1bbde584b4921,
                    0x6ba8e8f303d6dcd4,
                    0x001c82a1469874b8,
                ])),
                Felt::new(BigInteger256([
                    0xaa6f2082a4d327e3,
                    0x2ded4314d8f777ef,
                    0x1053ec0c2b5a77e2,
                    0x01d4ddd3e0d2e1b8,
                ])),
                Felt::new(BigInteger256([
                    0xee81855a4afe7df0,
                    0x3b5c27176a2658e9,
                    0xc1eb304d9ac6fb9a,
                    0x127ba65ada61fc15,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xfa6f85346feedf2a,
                0xbbbfd652692b2690,
                0x34481f7ef7ca769c,
                0x09424b4cec000e9c,
            ]))],
            [Felt::new(BigInteger256([
                0xc36754f27c8ad864,
                0x7585a97591f93471,
                0xb24254578f6c5447,
                0x0361748b8c160a1c,
            ]))],
            [Felt::new(BigInteger256([
                0x05a21f0e964c29c7,
                0xc28ceeb2dcae3e63,
                0x2b99e57f734fb193,
                0x0edf2ee981c2c6f7,
            ]))],
            [Felt::new(BigInteger256([
                0x5b2f3e60f5c7bca8,
                0x831c672916fbc0e7,
                0x337759b61cd2da25,
                0x02b25d3657c73aef,
            ]))],
            [Felt::new(BigInteger256([
                0x51aadc72ec25806e,
                0xc12a4ab691a5c18a,
                0xc4e948f09b162e83,
                0x1259d51658115367,
            ]))],
            [Felt::new(BigInteger256([
                0xd388152dcbdaa7c8,
                0xf4107fc081f243c4,
                0xcbb8092c8df3caef,
                0x0be732224145c4bd,
            ]))],
            [Felt::new(BigInteger256([
                0x68ff25c8d1aeb4ed,
                0x87441d26003d4e56,
                0xc29231a7c45df539,
                0x0e124c548421fd03,
            ]))],
            [Felt::new(BigInteger256([
                0x4e630db931913968,
                0xa34a851470489859,
                0x8cb97d5ea6a598a4,
                0x0b1d7a7c63cf8a2a,
            ]))],
            [Felt::new(BigInteger256([
                0x02ba70afcbef9506,
                0x364249c54fba5aff,
                0xa9189338cde769fc,
                0x00585329b4503498,
            ]))],
            [Felt::new(BigInteger256([
                0xc37c87e43572c02d,
                0xe8e84354cfda4a05,
                0x6a741960445edc95,
                0x0411f1b81fdd7205,
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
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xfa6f85346feedf2a,
                0xbbbfd652692b2690,
                0x34481f7ef7ca769c,
                0x09424b4cec000e9c,
            ]))],
            [Felt::new(BigInteger256([
                0xc36754f27c8ad864,
                0x7585a97591f93471,
                0xb24254578f6c5447,
                0x0361748b8c160a1c,
            ]))],
            [Felt::new(BigInteger256([
                0x05a21f0e964c29c7,
                0xc28ceeb2dcae3e63,
                0x2b99e57f734fb193,
                0x0edf2ee981c2c6f7,
            ]))],
            [Felt::new(BigInteger256([
                0x5b2f3e60f5c7bca8,
                0x831c672916fbc0e7,
                0x337759b61cd2da25,
                0x02b25d3657c73aef,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 248];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);
            bytes[124..155].copy_from_slice(&to_bytes!(input[4]).unwrap()[0..31]);
            bytes[155..186].copy_from_slice(&to_bytes!(input[5]).unwrap()[0..31]);
            bytes[186..217].copy_from_slice(&to_bytes!(input[6]).unwrap()[0..31]);
            bytes[217..248].copy_from_slice(&to_bytes!(input[7]).unwrap()[0..31]);
            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
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
                    0x8c56e3c71101ba1e,
                    0x53172ea2590c9d47,
                    0xf1c25581a2e28f98,
                    0x06bbc3a22a5a34e0,
                ])),
                Felt::new(BigInteger256([
                    0xf2020c11cad5d7f5,
                    0xee4c414de8aa6010,
                    0x18140220e4203936,
                    0x105f362f9a29a4b1,
                ])),
                Felt::new(BigInteger256([
                    0x9ba70bc6a5dfc70f,
                    0x563aa9c76763ef4c,
                    0xc7aaf1393353aad3,
                    0x0262a852cd9e9e70,
                ])),
                Felt::new(BigInteger256([
                    0xeee930881cac2ce8,
                    0x34d34b97525eece8,
                    0x7a575cd36f999dcb,
                    0x02273bbb69c82996,
                ])),
                Felt::new(BigInteger256([
                    0x9109b2ad2c0cfbce,
                    0x635d1eb2276b0e02,
                    0x870a98707f3fbb0b,
                    0x0b6963b92f8a78c4,
                ])),
                Felt::new(BigInteger256([
                    0xf558d245eccbabe2,
                    0xfbc35d84c551feab,
                    0x0171358894a013e1,
                    0x0c57f065298ae64a,
                ])),
                Felt::new(BigInteger256([
                    0x4acec451ce9aaf3d,
                    0x7c51c4ba67c39b35,
                    0xf244cee498c21f62,
                    0x03b50be8956c8c42,
                ])),
                Felt::new(BigInteger256([
                    0x70524584d8fdd3ad,
                    0xdcea511060a53b19,
                    0x96a1ad95951cdadd,
                    0x0e964c92a05de24c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6ad06e549d22074e,
                    0xfcc9dc1ee49aa3cc,
                    0xe1e43a80b78c7374,
                    0x0bc9e3b747711a29,
                ])),
                Felt::new(BigInteger256([
                    0xf117cb50bf2c4a29,
                    0x2ed911fd4cc906c1,
                    0x116b31c158c122b9,
                    0x0636587fbaf10c79,
                ])),
                Felt::new(BigInteger256([
                    0x69765571c22787ac,
                    0x4d50f0dab47d9993,
                    0xe3c756d3d5c639f3,
                    0x0fd6722cc96b84b6,
                ])),
                Felt::new(BigInteger256([
                    0x6a8ffa0dfa5b7864,
                    0xc7533c6de1b4ffee,
                    0xe5b71e5d4fd5a42e,
                    0x00078599cfe0fd08,
                ])),
                Felt::new(BigInteger256([
                    0x4618175e4cbceb3d,
                    0x5696d46b528f0e8b,
                    0x9c93171136534138,
                    0x0b930f9ac896a70f,
                ])),
                Felt::new(BigInteger256([
                    0x97746ef11fadd441,
                    0x5459c9bd1f31836d,
                    0x101b13ff329a6c81,
                    0x0ca2b0ca131833d0,
                ])),
                Felt::new(BigInteger256([
                    0xbad33d382633e744,
                    0x907198a25d888f3e,
                    0x68a939ca34464e42,
                    0x0fe1ad166c16478d,
                ])),
                Felt::new(BigInteger256([
                    0x5cae8d7324cceb41,
                    0xf0326b9e0f9aec88,
                    0x29c98bbc69ad1aa6,
                    0x08cd5d6cc4f85f68,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x75d67121f199a5ad,
                    0xef63f22704caf49b,
                    0x7028d04df1066e55,
                    0x112fddd3cf001255,
                ])),
                Felt::new(BigInteger256([
                    0x4f0a383dc2a4d488,
                    0x49f04920cdd1be52,
                    0xdc97e680c23560a4,
                    0x03c0b39a17cc26ec,
                ])),
                Felt::new(BigInteger256([
                    0x861a72510002127c,
                    0x69d7d55ae9647056,
                    0x4d0980ee622b0ddb,
                    0x1201a10e420386a6,
                ])),
                Felt::new(BigInteger256([
                    0xf3e4e63d35fd9f84,
                    0xcfedc333bce42e9d,
                    0xf8d95bba650557c6,
                    0x060d2a314c417847,
                ])),
                Felt::new(BigInteger256([
                    0x6cb15a1c32e6e565,
                    0x53a5e483fd241e52,
                    0x4b3fb1688299e6ab,
                    0x0d30f3b5af7a7772,
                ])),
                Felt::new(BigInteger256([
                    0x5b7cb38cbfe00bd1,
                    0x293660c9d4886982,
                    0x063fde687ebc1124,
                    0x077248175e1d0185,
                ])),
                Felt::new(BigInteger256([
                    0x85a19e573da12db4,
                    0x86fba8959952431a,
                    0x5fca387e67e0ad15,
                    0x0d87908c26a11de3,
                ])),
                Felt::new(BigInteger256([
                    0xc6fa13f1d41d036c,
                    0x12b51cc3507837ee,
                    0x4c6cebd1c69683e8,
                    0x0011b7302e352790,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5955f27a75f1a730,
                    0x643e461f56e21cf2,
                    0x06cf4556a7ee54e7,
                    0x080228e953b97011,
                ])),
                Felt::new(BigInteger256([
                    0xfaeb3b3961caa932,
                    0x6073ec269fc84634,
                    0xea81f2565b5f64b8,
                    0x1032aab71dcc660e,
                ])),
                Felt::new(BigInteger256([
                    0x9c2c53a1b8bf936b,
                    0xfcfb75d087a72406,
                    0x8d218f21c2a4ba83,
                    0x0cdd2c0b43fc3c06,
                ])),
                Felt::new(BigInteger256([
                    0xc85f98d32cb2df80,
                    0x3b27707d905096c0,
                    0x58b3ad24b302c2f2,
                    0x040a9b48b011ed78,
                ])),
                Felt::new(BigInteger256([
                    0x28bc2a73aefae199,
                    0xb0f92253a7c086d3,
                    0x4f543151c56fad6a,
                    0x0def50398e88e72e,
                ])),
                Felt::new(BigInteger256([
                    0xa45e302b189f371c,
                    0xcfc90f69813316cc,
                    0xa3d185fb0291cdab,
                    0x077c6e9b0e3fe30d,
                ])),
                Felt::new(BigInteger256([
                    0x406f235faca3479e,
                    0xc9521e41e13f9056,
                    0x752e6d896a7f0ab1,
                    0x049e6d9cecfa2158,
                ])),
                Felt::new(BigInteger256([
                    0x3f5192d8a886a208,
                    0xd62a817d91b61c8a,
                    0x69d6ea6d621125e4,
                    0x0383d05e0346ccee,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x979f6e682178ea01,
                    0x5bb4c5b1f8779e30,
                    0xdec775b2fc22d89f,
                    0x0cb28bb7a1276367,
                ])),
                Felt::new(BigInteger256([
                    0x8bdce5f309217427,
                    0x827d75e6283282bc,
                    0x53f1c633db66666e,
                    0x0fe31175c6509f24,
                ])),
                Felt::new(BigInteger256([
                    0xeae412e02fa9cab3,
                    0xeb4fbb41df6f56c5,
                    0xc8ad2e34cb1518b8,
                    0x0f4ca1d6e4396e95,
                ])),
                Felt::new(BigInteger256([
                    0x66b00a9dafa29157,
                    0x147fdb728ddfd818,
                    0xed2b95de89350abb,
                    0x07536199f7f39ae0,
                ])),
                Felt::new(BigInteger256([
                    0xbad38976e33390e8,
                    0x4fced7dc9fc5375d,
                    0xd5aef2ae879a32e1,
                    0x1013775b2612deec,
                ])),
                Felt::new(BigInteger256([
                    0x4d35c832a2cbe93a,
                    0xb3e7852a563213ff,
                    0xd7b8b5dc758eb5d2,
                    0x0ed02ee2d5719afa,
                ])),
                Felt::new(BigInteger256([
                    0xad38b39152484d12,
                    0x4b50ae9e248e6fa0,
                    0xbda6ce7b6e6a8b73,
                    0x12a0eaa5e1e5b4e6,
                ])),
                Felt::new(BigInteger256([
                    0x1e012a4062ec4608,
                    0xbe0016977afe8f4d,
                    0x884d736f2f6925c7,
                    0x105556358b8e64fe,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaf8cd15a7a586839,
                    0x8f31785152dc56f3,
                    0x2bbb7ee59e9b29eb,
                    0x10595c4467b4de53,
                ])),
                Felt::new(BigInteger256([
                    0x9b118d73762508fd,
                    0xec40def97cce2846,
                    0x1d2db11ef86f9888,
                    0x08412559d59cf6a2,
                ])),
                Felt::new(BigInteger256([
                    0xa0befd1f22ae0efd,
                    0x30474448a935946f,
                    0xa2ee92a900fd96b4,
                    0x0883f7e25e4d5f9c,
                ])),
                Felt::new(BigInteger256([
                    0x6792d33a93964141,
                    0xb2e936b557e8430c,
                    0x87d2f7e37bc847b3,
                    0x0d26e714b5e0636e,
                ])),
                Felt::new(BigInteger256([
                    0xe2981966033cf09c,
                    0x20752f29c8b12583,
                    0xa442fcb0eaa1a60d,
                    0x0e7c8179b51d5605,
                ])),
                Felt::new(BigInteger256([
                    0x6f6d5089662164fe,
                    0x9c91c2429372732b,
                    0x0b08d5b248815d64,
                    0x017a4be3d931d013,
                ])),
                Felt::new(BigInteger256([
                    0x66e9593a2880e92a,
                    0xf82a2d48c74beff5,
                    0x81f66ed0eb5ea73b,
                    0x104463d9bd631d22,
                ])),
                Felt::new(BigInteger256([
                    0xe47bc259193d59fa,
                    0x6c2f4012c933c2e5,
                    0x083df36f72eb2940,
                    0x0adfce39cf9ac2e7,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x4a5acdab77af7d38,
                    0x709fd9e24620d2f9,
                    0xccc72d3f8b35da9c,
                    0x045b0493a325e419,
                ])),
                Felt::new(BigInteger256([
                    0xbbeafd032214271c,
                    0x7979d86c39086528,
                    0x184b5db24881b590,
                    0x0bae16a153f9b1f6,
                ])),
                Felt::new(BigInteger256([
                    0x6b6b03d01eb3511c,
                    0xf2764beb0c0d6300,
                    0x5d0e5631e9c0155b,
                    0x0c15693b975e40e9,
                ])),
                Felt::new(BigInteger256([
                    0x7304f53ae947fb6a,
                    0xff81ead900d85ebb,
                    0x10b8e3c82d62ed01,
                    0x0f3e459f63a3c974,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd7ca5722fb96df31,
                    0x4a242799e597119c,
                    0x6ff56d182ba9a360,
                    0x065bdb368c47fff5,
                ])),
                Felt::new(BigInteger256([
                    0xddfd477021c5a226,
                    0x4a2ebc96e3f444f7,
                    0x19e8021da9550e75,
                    0x03e374dfe9ce8ade,
                ])),
                Felt::new(BigInteger256([
                    0x9daf86382678005c,
                    0x9bc41829ea834a1d,
                    0x98c39bdab4d2e1d4,
                    0x02190f780f8b2bb9,
                ])),
                Felt::new(BigInteger256([
                    0x3dd31d2b51c74003,
                    0x21fdc75621bd62bb,
                    0x9fb1a65d522abdb4,
                    0x051c302579652203,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x603030324d7856bf,
                    0x5302a2bb093c2f5b,
                    0x44fe144e64c707e0,
                    0x096489fc178919a8,
                ])),
                Felt::new(BigInteger256([
                    0xcc080fc932a83f5e,
                    0x5ce713ecd5c9fda6,
                    0x8f035080fb6b5aa0,
                    0x0a77f109104e9544,
                ])),
                Felt::new(BigInteger256([
                    0x9435e3a2bc3b3bd7,
                    0x67b3349b81d219dd,
                    0x8dddc2ec78292fc9,
                    0x0bbe760e95f81534,
                ])),
                Felt::new(BigInteger256([
                    0x794767df393d97c8,
                    0xec16e1c06ef856aa,
                    0x84b4a58049b5eadb,
                    0x0499e148f22b05ef,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb4d745faa674872b,
                    0xfc2dd42c391cbfe4,
                    0xf600062ed0dec191,
                    0x085658b74445b705,
                ])),
                Felt::new(BigInteger256([
                    0x4d9b5d72c44be6d7,
                    0x582fcc30bee712a4,
                    0xabe7f30ed9f85bc4,
                    0x0efa63f44db2b6c7,
                ])),
                Felt::new(BigInteger256([
                    0xaf1ac62a55a213e8,
                    0x613d76fad0d29941,
                    0xcc6bc3c3ec4112fe,
                    0x0513c3be52756e77,
                ])),
                Felt::new(BigInteger256([
                    0x3fde0187ac7c983a,
                    0xc0eb7efbca27d078,
                    0x1427a1eaeeae0120,
                    0x076c6f05e6394ca5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdd020dfaa2d3566b,
                    0x18f0b64c7bd1640b,
                    0x3f261af2f7b40db9,
                    0x044185a5e92bbfc9,
                ])),
                Felt::new(BigInteger256([
                    0xac8ba0a42c06cc92,
                    0x0d12ab6df62c4626,
                    0xdab5e6602b55ae03,
                    0x07cae0ded3c13504,
                ])),
                Felt::new(BigInteger256([
                    0xc9a8ab37f01cfa48,
                    0x63e0ed0a5beea07f,
                    0x8c3a3bbbf6fa18ff,
                    0x0aad75db1fc23dd0,
                ])),
                Felt::new(BigInteger256([
                    0x58be4a1fb67902d9,
                    0xb74c146331eb00fe,
                    0xda372591c80dd91b,
                    0x0ca10d14388dae30,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf88472ce5a75695a,
                    0xce74260153617b01,
                    0xabfeb9327799a519,
                    0x0ab358f23f486bcc,
                ])),
                Felt::new(BigInteger256([
                    0xeb3867244530919a,
                    0x7e3d80544442902f,
                    0xfffdcb1fd16b9dc8,
                    0x00717032b40aa43e,
                ])),
                Felt::new(BigInteger256([
                    0xa47cba5e6a051646,
                    0x39d76499b4d5ef19,
                    0x0a3689fa8c6fb469,
                    0x0f093e8d0f22affe,
                ])),
                Felt::new(BigInteger256([
                    0x3545125fe321ed8c,
                    0x38b4d8665158b805,
                    0x3a2c0e33eeccfa4c,
                    0x0f7702aa26e49fe3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc4e7f5231402adad,
                    0x4d5d3199bb7a5938,
                    0x1180d3647298bc0b,
                    0x0dc15657ce1a2602,
                ])),
                Felt::new(BigInteger256([
                    0xf0f9886ab326e110,
                    0x504ba3d0b0ceb0a8,
                    0xfdd1d486aca1e188,
                    0x098ba637f76c5cf6,
                ])),
                Felt::new(BigInteger256([
                    0x6998fbee0d7a9f95,
                    0x80ad38a73e2de2fe,
                    0x96bd4561b23027d2,
                    0x021fa873e72e469c,
                ])),
                Felt::new(BigInteger256([
                    0x2bcc3f4d002e6eb2,
                    0x2a08819ec99e7178,
                    0x590f2f5c37451ac2,
                    0x11857c5d08d413ba,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x39b59d6565bc3e9f,
                    0x3e3d0bce6d72189f,
                    0x8b569f06380b64dd,
                    0x0004335e74026bab,
                ])),
                Felt::new(BigInteger256([
                    0xa991a7698acd93af,
                    0xbc275e7bb9ec379e,
                    0x8586cecad5c580d2,
                    0x0423ffef1d5f5937,
                ])),
                Felt::new(BigInteger256([
                    0xc0d30c3b7d59d0d7,
                    0x834ece284c24953d,
                    0xf7a42591fd2cbebc,
                    0x0f405778dfc37494,
                ])),
                Felt::new(BigInteger256([
                    0x14fb6452c287c328,
                    0xf66d3f95f6df44c1,
                    0xd591abec932f69a8,
                    0x126bfea858c8ae23,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeffe9c35c0f66c1f,
                    0xa0b78a09bede2603,
                    0x52fc52d163a81f9a,
                    0x07916240f64b560d,
                ])),
                Felt::new(BigInteger256([
                    0x597b42aecfab97da,
                    0x83619d1ae58cd838,
                    0x561e7a50f1f68a7d,
                    0x04d3382ebaf3b596,
                ])),
                Felt::new(BigInteger256([
                    0x4ccfbdd643dbe101,
                    0x8fe82800300d143a,
                    0xb0e857a5d1d10220,
                    0x0ce2b8cba63a8458,
                ])),
                Felt::new(BigInteger256([
                    0x36812e13afeacb5a,
                    0x03a90086c1d2b216,
                    0xfe1613a492f87cc3,
                    0x0ec74ca7926a5102,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd1ed2af4f9b4ea1b,
                    0x96d876d742d83a8a,
                    0x4d83340839c8fbb0,
                    0x0e4a3e0d838ba673,
                ])),
                Felt::new(BigInteger256([
                    0x3f47bf0902fc5470,
                    0x6440a8ae08ab8bfc,
                    0x8c85843b23ede2be,
                    0x0c3d1345d0cc2fc5,
                ])),
                Felt::new(BigInteger256([
                    0x01834b9cd3f18a95,
                    0xf96f943e3bca2838,
                    0x1269ffcc59760ddc,
                    0x119be5a413c768aa,
                ])),
                Felt::new(BigInteger256([
                    0x64adfdb8eed3c958,
                    0xa087625d275fdc39,
                    0x28b19df52b24d348,
                    0x038699dba2c47f67,
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
                    0x1ea6dcf99bdc1410,
                    0x2d04054be551b370,
                    0xe20694bfbd393bd4,
                    0x03520c9652b0cf0b,
                ])),
                Felt::new(BigInteger256([
                    0x846013e7e6042e82,
                    0x925bbbe062db7ee6,
                    0x3f316891ebf986e4,
                    0x1076f6bac6ea749a,
                ])),
                Felt::new(BigInteger256([
                    0xa58f17b23dd5b227,
                    0x9510b75feb77173b,
                    0xbb8c77a943a0f729,
                    0x05559c667ee50a7c,
                ])),
                Felt::new(BigInteger256([
                    0x8b8a1035363f115b,
                    0xdb29afe2d9ee3bfb,
                    0xd5715baf85711914,
                    0x0c397edde73e5e24,
                ])),
                Felt::new(BigInteger256([
                    0x6fa84a4d0e437a12,
                    0xdf2648ef55f08a8b,
                    0x5e7cb55b43a79092,
                    0x0a0644794332012c,
                ])),
                Felt::new(BigInteger256([
                    0x556418b3daffe6b4,
                    0xfe959b81c46b8b0c,
                    0x474472de82895797,
                    0x0d6ed9978cf5fa5e,
                ])),
                Felt::new(BigInteger256([
                    0x62936b9ead8bd1af,
                    0x5efa4a36ac1aee8a,
                    0xd2bbcc97cd75515b,
                    0x08f34ba0e04afa0b,
                ])),
                Felt::new(BigInteger256([
                    0xb40c99a72ad4aff4,
                    0xd0b94c3a22e9dba8,
                    0x0e3aa0a5b4c9244c,
                    0x013bb95709fdbdb1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7936c55917ef41d3,
                    0xf4607138a44513f9,
                    0xc15ff8ba7f4c6ac5,
                    0x12667a7ead702e91,
                ])),
                Felt::new(BigInteger256([
                    0x4c05fa72e7d257b2,
                    0xf0767a6e9581abfb,
                    0x25f713ca6099f2bc,
                    0x0d5f40d863747e11,
                ])),
                Felt::new(BigInteger256([
                    0x5e6a96a02544e998,
                    0xce3503ce6078b0c3,
                    0xc34b3925f97e65b8,
                    0x06298c1631b96c50,
                ])),
                Felt::new(BigInteger256([
                    0xc25aa8aa1c759b5e,
                    0x3ec5042574829e91,
                    0x29f6f49e03ebd6ff,
                    0x012145548b46f29f,
                ])),
                Felt::new(BigInteger256([
                    0xac502f5ebd4fa21c,
                    0xf26c9e9c08b3daa1,
                    0xd590d5474bb09816,
                    0x00ca0c81dcd64e48,
                ])),
                Felt::new(BigInteger256([
                    0x932f8c093c94d031,
                    0x87a399eb1eceba92,
                    0x1206b1d794adaaed,
                    0x0a9c0dd74eb48c89,
                ])),
                Felt::new(BigInteger256([
                    0x9ec68b9031f97bf3,
                    0x5731b6149b8b2622,
                    0x93e13f3d76c21d81,
                    0x0e91edcbd1a65aa0,
                ])),
                Felt::new(BigInteger256([
                    0x6eb773f9fd265a9e,
                    0x13d0aafcd8022908,
                    0x023f5f88b1b6ba4c,
                    0x06a270298362d6f2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x363ac1004d3ff041,
                    0x99e22f8fa216fffb,
                    0xbdd91d571f524f9e,
                    0x00c2e3cd2f9bc013,
                ])),
                Felt::new(BigInteger256([
                    0xb24ad12be99961f1,
                    0x95d38c5bc01e9681,
                    0x9f4b1cca35a73c5f,
                    0x02be238b27709c2b,
                ])),
                Felt::new(BigInteger256([
                    0x857901178ab8d289,
                    0x5a4e47c29a377f73,
                    0x9b1838c9eab29586,
                    0x0528aa15d240eb28,
                ])),
                Felt::new(BigInteger256([
                    0xb3f3baaa82e0c164,
                    0x8604aaaf05d9a261,
                    0x9a6314b8d52a9a2c,
                    0x070af5f26997ce29,
                ])),
                Felt::new(BigInteger256([
                    0x799742891cd6afae,
                    0x7e973da6287a0217,
                    0x1a22c598a6e52c0c,
                    0x0fc8c2cdcc5e94fe,
                ])),
                Felt::new(BigInteger256([
                    0x10f82bf655187270,
                    0x7e0b393efe7d7524,
                    0x7a9c9d21857f3d32,
                    0x046fe78d6017a367,
                ])),
                Felt::new(BigInteger256([
                    0xb58240865612477a,
                    0x8758437d5d1d5ef0,
                    0x1c574d4b64e5ce64,
                    0x0a83b694f00af83c,
                ])),
                Felt::new(BigInteger256([
                    0x9c1b349e8dfa1894,
                    0x06816a17ea26c6cc,
                    0x8bf9b8510b766cdf,
                    0x0014356a19f9ca3a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9859b35992a26e68,
                    0x9621eaac58b0c060,
                    0x3e3105bf061b84e9,
                    0x0deee0c617f5b566,
                ])),
                Felt::new(BigInteger256([
                    0xbdf5d59dfd5bdd92,
                    0x1dd5cb170c6e6d3f,
                    0x07dec1c93dc88b24,
                    0x0b32ea96df57ef61,
                ])),
                Felt::new(BigInteger256([
                    0x4ea63b2d69d538e5,
                    0xd997cef9c0115fbc,
                    0x74f19ff9ec5ca2ce,
                    0x126397051be56a97,
                ])),
                Felt::new(BigInteger256([
                    0x8f363ba3ef23d407,
                    0x463430d3aeaf7d72,
                    0xf31165e86302e65f,
                    0x074b2adc5fc325c8,
                ])),
                Felt::new(BigInteger256([
                    0x3143d67996886b49,
                    0x92d214a5060ab432,
                    0x76cb5b77af726743,
                    0x06de7ee2f7b996f6,
                ])),
                Felt::new(BigInteger256([
                    0xe85c5a27865cff0d,
                    0xb916ad3c1f0d9890,
                    0x1e987af51949413b,
                    0x032f2ea095e53c1e,
                ])),
                Felt::new(BigInteger256([
                    0x0a57fdede639c8dc,
                    0x20d25923f575e4f7,
                    0x1de4242bbd999ec4,
                    0x125c076a35bed148,
                ])),
                Felt::new(BigInteger256([
                    0x65a22c31f46a5868,
                    0x7b3eb093b8a7aa41,
                    0x79c412664fc953d1,
                    0x0942e1de2e43f597,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9b190da485f98c27,
                    0x1dfec435503f76c7,
                    0xc9f541bf4d6be356,
                    0x0aa6f43aa4581153,
                ])),
                Felt::new(BigInteger256([
                    0xa80c5d095c96c6ff,
                    0x4cf14e8d0f9e1cbb,
                    0x70d9db56f65e0016,
                    0x06ce46de843e35bb,
                ])),
                Felt::new(BigInteger256([
                    0xf6e183d79337fbeb,
                    0xcf51270adbd27c04,
                    0xab7a7288e867fd8b,
                    0x0bfff4e702c60fa8,
                ])),
                Felt::new(BigInteger256([
                    0x678f27e552662404,
                    0x91a0e35db1b68377,
                    0xcda8665ee831db2b,
                    0x0b29e62f52f3474e,
                ])),
                Felt::new(BigInteger256([
                    0xfa901d53f5549bb3,
                    0x106c896395e2842a,
                    0xb851840f72b20f57,
                    0x12851c268e77f76b,
                ])),
                Felt::new(BigInteger256([
                    0x897fe621c6096868,
                    0x37d584c427ae054c,
                    0x4e617c0164daf341,
                    0x117ce82df8a81078,
                ])),
                Felt::new(BigInteger256([
                    0xcade6cffbbb73432,
                    0xf35b094aa86a82be,
                    0x3b618dc9464442dc,
                    0x0e21f974df4ddb76,
                ])),
                Felt::new(BigInteger256([
                    0x05e8999a13ecb89e,
                    0x27952bc96e71a7ce,
                    0xfdff24d710346fb9,
                    0x0e79d201a7e65155,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1739930b2ca21977,
                    0x0003cc82c08a42d1,
                    0x43fbd6dc501ca709,
                    0x09b75f9729ab876e,
                ])),
                Felt::new(BigInteger256([
                    0x6a5f74a9395244f5,
                    0x2c8dc5b1bd333e8f,
                    0x3f1f5456aa1337d4,
                    0x0c0753db39c1adb8,
                ])),
                Felt::new(BigInteger256([
                    0x8a1bcb01aee6f4f1,
                    0x36f8dbc300018efc,
                    0x6e6e49bf9e11f91f,
                    0x022a5ffb37f9ef3f,
                ])),
                Felt::new(BigInteger256([
                    0x97de59258e35c840,
                    0xfb685141ebc4c306,
                    0x7adfa03591f43279,
                    0x0215ff51d2627b24,
                ])),
                Felt::new(BigInteger256([
                    0x4a08b028ec5d9df7,
                    0x2820f00efaccc9ec,
                    0xf5fc7efa82ae279d,
                    0x06df22bc4aa90155,
                ])),
                Felt::new(BigInteger256([
                    0x99f1edf87a17865d,
                    0x00e8813a55e11709,
                    0x24bb5ee26952ecce,
                    0x078a3501b8905aec,
                ])),
                Felt::new(BigInteger256([
                    0x7fbf0a480189681a,
                    0x438b6dd99ff0e33a,
                    0x9628e9af7acd6b50,
                    0x05c51b39748027ea,
                ])),
                Felt::new(BigInteger256([
                    0xd769b1032dbe47ff,
                    0xb3e2900a3c16a09f,
                    0x2462a41a08ad73c3,
                    0x11f0dd414907a933,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xd092c3b9a1bef0d8,
                0x28bcfb14ec0ef9db,
                0x91712aaf326b3288,
                0x0605ff52bdc855c0,
            ]))],
            [Felt::new(BigInteger256([
                0x914a41f6959bc1b6,
                0x5214c3b0d5cc036d,
                0xc252b16ddbfc515e,
                0x11748fb3ff06d890,
            ]))],
            [Felt::new(BigInteger256([
                0x2fa40b7d759969bb,
                0xaa095604ffd09d89,
                0x85df801dc5d9cd24,
                0x11896cfe15ce24ba,
            ]))],
            [Felt::new(BigInteger256([
                0xe759eb1f6cdf1a23,
                0x1cdc1f54c2fe3c41,
                0x21c711ce298e8174,
                0x11258a11307a8394,
            ]))],
            [Felt::new(BigInteger256([
                0x507305b88450d863,
                0x1b0c5103a8d5de2b,
                0xfe4121b8c9900330,
                0x093a0ddcbb677d9d,
            ]))],
            [Felt::new(BigInteger256([
                0x49a972ba76e9ac45,
                0x44764b7e602efc20,
                0x3fb4717e4c6e3418,
                0x022f6beb7f48c433,
            ]))],
            [Felt::new(BigInteger256([
                0x3e37aeea7e2b21c8,
                0x0ed0b75492b97178,
                0xa334fc0a2f3bee4d,
                0x125b8f9f382bb450,
            ]))],
            [Felt::new(BigInteger256([
                0x5ad5794ecd060624,
                0x2f1f712bc07fdc54,
                0x56cf6b06c04711ef,
                0x04549a9538700401,
            ]))],
            [Felt::new(BigInteger256([
                0x6e1e015ae3d80b5f,
                0x196f0eb6c4a84a35,
                0xad8bd7456176e83a,
                0x0c1eaeb8c7d2f7c2,
            ]))],
            [Felt::new(BigInteger256([
                0x21d291aea11d5bb8,
                0x380b1a0ba09447df,
                0x6087be33aa226a5e,
                0x0629de0d71d7b8b6,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
