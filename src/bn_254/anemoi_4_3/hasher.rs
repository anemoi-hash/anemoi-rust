//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::bn_254::anemoi_4_3::AnemoiBn254_4_3;
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiBn254_4_3 {
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
                AnemoiBn254_4_3::permutation(&mut state);
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
            AnemoiBn254_4_3::permutation(&mut state);
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
                AnemoiBn254_4_3::permutation(&mut state);
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
            AnemoiBn254_4_3::permutation(&mut state);
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
        AnemoiBn254_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiBn254_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiBn254_4_3::permutation(&mut state);

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
        AnemoiBn254_4_3::permutation(&mut state);

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
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x873aa36b23807561,
                0xb14d0386dba5173d,
                0xb57317e70078fead,
                0x1522ecae0f9de3d3,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xa712fcdb1e0f455b,
                    0xc4e77420087d371e,
                    0xbad23613b11f1ed9,
                    0x24024ec065eede03,
                ])),
                Felt::new(BigInteger256([
                    0x1ba9476bc5b0f2fd,
                    0x050c1a7bcf3e4842,
                    0x174c67ca631bb313,
                    0x1f68fd2a180fc4b4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe0b483c21f35f730,
                    0x84382444a94218fe,
                    0xc517056818d1ed7e,
                    0x3024ad46073d5ba1,
                ])),
                Felt::new(BigInteger256([
                    0xa79e0ab39ae6e015,
                    0xc6baacb2df5c3f94,
                    0x7754d51a6178c8bf,
                    0x1e7de524b457860c,
                ])),
                Felt::new(BigInteger256([
                    0x4fd6bb3ba1d9b5e4,
                    0x84d0b383d8777a19,
                    0xd89de0fb47e5cb3d,
                    0x15bebf1db91f1859,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x498e23ef14f4c3f4,
                    0xf2b0d785d8e117f2,
                    0x313a18828310f149,
                    0x115fef950dfbb39d,
                ])),
                Felt::new(BigInteger256([
                    0xdb0e6e3f93682d8b,
                    0x0d18387226021cae,
                    0xfbf8001c9d31beae,
                    0x24ee73183e31e91a,
                ])),
                Felt::new(BigInteger256([
                    0xd8af3f4cdfc4ad00,
                    0x3b37babdb37b1dca,
                    0xd4b043c7ea1b58c7,
                    0x0c8fee4d113f107a,
                ])),
                Felt::new(BigInteger256([
                    0x6c33f17248c44ab5,
                    0x688a0a8c043facad,
                    0x3be46531ff6ff95a,
                    0x1fbcfe26c8b9b96a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb24d1b5f3f6f2dce,
                    0xac6e7db258bd56fe,
                    0x1053d0ad0b464269,
                    0x19012c8ce5cc00ff,
                ])),
                Felt::new(BigInteger256([
                    0xd5cfe2f4edd161a1,
                    0x08a92b9891ba6d20,
                    0x9a3cff6683d063d0,
                    0x0aa828312060a72e,
                ])),
                Felt::new(BigInteger256([
                    0xa81e4ea7d9d6209d,
                    0x6e289da6d6c82df5,
                    0x65638a5101180b3d,
                    0x04a49a49d86bfa86,
                ])),
                Felt::new(BigInteger256([
                    0x60107283f5c296a6,
                    0x62e79a4e02b641c1,
                    0x2483a88d71904d3d,
                    0x136af7452418806d,
                ])),
                Felt::new(BigInteger256([
                    0x9b890ed26771d8cb,
                    0x80322d33021aa518,
                    0x920a3c7ce3d3c195,
                    0x011d99748d7838ba,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6d9b4421f3b62bd6,
                    0xaf0cd614fa15fb8a,
                    0x274fafa3302b3cd7,
                    0x07c20a87ece94759,
                ])),
                Felt::new(BigInteger256([
                    0x491abd3c9cab778f,
                    0x57d3f95dc6c0857d,
                    0x6ea62a45470826a6,
                    0x20b82cea425b2c58,
                ])),
                Felt::new(BigInteger256([
                    0x051c88d442a9541a,
                    0xce418da81f019325,
                    0x1dc1ad83951ec902,
                    0x1c57760f3c4127ea,
                ])),
                Felt::new(BigInteger256([
                    0x4260cb9db43c3527,
                    0xf261ce1289638c92,
                    0x58a754b44692eff1,
                    0x1628a7a184574c8c,
                ])),
                Felt::new(BigInteger256([
                    0x7fffb4a3512efa6b,
                    0x554e65eb4c97b9a2,
                    0x5523290932a16e66,
                    0x2611f664c4e4c50b,
                ])),
                Felt::new(BigInteger256([
                    0x16eb97e782a8cfe3,
                    0x6d8a5001defbd2b3,
                    0xf07fe71a2aa935bc,
                    0x02bc06f8cb11ae56,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x045964e79dff403e,
                0xf0449197558aa720,
                0xb6e04e20a3f20171,
                0x060eab906bdc8bee,
            ]))],
            [Felt::new(BigInteger256([
                0x6109c0ef173cb793,
                0x231287edce74f400,
                0x02a874fca06b1a86,
                0x07b2b6281e312462,
            ]))],
            [Felt::new(BigInteger256([
                0xc0bbc6a27f201888,
                0xa03a358163c0d0b9,
                0xe5a019dd3529c9c8,
                0x0ff04487a88c39df,
            ]))],
            [Felt::new(BigInteger256([
                0x79e410c37b8376d8,
                0x94bcdbf73736f010,
                0xb7e8e05b2672e53e,
                0x27f610549fe34706,
            ]))],
            [Felt::new(BigInteger256([
                0x17ffdb53c4f30493,
                0x9d1f5a3f7e66b33f,
                0x240d9dc7f94a3a9a,
                0x06df82e331a044a1,
            ]))],
            [Felt::new(BigInteger256([
                0x3be92044e72b61ce,
                0xe64435aba59e3e4e,
                0x8055a560c564f5d8,
                0x1418f3eea6f09547,
            ]))],
            [Felt::new(BigInteger256([
                0x1ec6517f032f84a1,
                0x3a0ad0f29a2bed95,
                0x8f3ea66285f08539,
                0x2f9475f3c008110d,
            ]))],
            [Felt::new(BigInteger256([
                0xf4230df7f6ae1872,
                0xe44110309e9f4949,
                0xa0bf6dc991220a78,
                0x2a62de8d60a9fad7,
            ]))],
            [Felt::new(BigInteger256([
                0x2d8de916c005de40,
                0xcf3cf7829456850a,
                0x64d08c739807d0e9,
                0x08d5c147faacf172,
            ]))],
            [Felt::new(BigInteger256([
                0x53fd51a54d12e1c8,
                0xbb12c42cbc133b4e,
                0x3ee548f595176eb5,
                0x1cd9f009bc723de7,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiBn254_4_3::hash_field(input).to_elements());
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
            [Felt::new(BigInteger256([
                0x045964e79dff403e,
                0xf0449197558aa720,
                0xb6e04e20a3f20171,
                0x060eab906bdc8bee,
            ]))],
            [Felt::new(BigInteger256([
                0x6109c0ef173cb793,
                0x231287edce74f400,
                0x02a874fca06b1a86,
                0x07b2b6281e312462,
            ]))],
            [Felt::new(BigInteger256([
                0xc0bbc6a27f201888,
                0xa03a358163c0d0b9,
                0xe5a019dd3529c9c8,
                0x0ff04487a88c39df,
            ]))],
            [Felt::new(BigInteger256([
                0x79e410c37b8376d8,
                0x94bcdbf73736f010,
                0xb7e8e05b2672e53e,
                0x27f610549fe34706,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 124];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiBn254_4_3::hash(&bytes).to_elements());
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
                Felt::new(BigInteger256([
                    0x63ae693eb8c60874,
                    0x3bf9a15fcc79894c,
                    0xef275e11c77a4c8e,
                    0x2e46df9b5c9e998a,
                ])),
                Felt::new(BigInteger256([
                    0xe13a207e043ee6ab,
                    0xfa083eab58d8b4f4,
                    0x66701cd45ca433e8,
                    0x2a8dbbaf274e047d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe2f549ca08371669,
                    0x2d0f24ff39b596dc,
                    0x21528bde4089b73e,
                    0x19fcab936113afdd,
                ])),
                Felt::new(BigInteger256([
                    0x69586a7e1522aad1,
                    0x452479b326037814,
                    0x983898b7ea3633ee,
                    0x08746d1ef5201d4f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa1b8d46d4cb1b486,
                    0xee8103d48e701356,
                    0xfd412da6510c1128,
                    0x011886310278c5a0,
                ])),
                Felt::new(BigInteger256([
                    0x6775d9ef32379394,
                    0x44f46aa251cdeea8,
                    0x8aec2293b76f8941,
                    0x25a0b882f63547db,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6e43f330e91fdb2f,
                    0xfcc474840e09cbe2,
                    0x331042e51d982954,
                    0x20548dbed7b42635,
                ])),
                Felt::new(BigInteger256([
                    0x26330e47d7247c2a,
                    0x3c307a640b6fe172,
                    0x8fd7f3186a162c52,
                    0x1510d2a8039969aa,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x08c7fda5e487f1d8,
                0x9e807579bce073b4,
                0x9d47352fa29d2819,
                0x28704cd7a2bafdde,
            ]))],
            [Felt::new(BigInteger256([
                0x4c4db4481d59c13a,
                0x72339eb25fb90ef1,
                0xb98b24962abfeb2c,
                0x227118b25633cd2c,
            ]))],
            [Felt::new(BigInteger256([
                0x092eae5c7ee9481a,
                0x33756e76e03e01ff,
                0x882d503a087b9a6a,
                0x26b93eb3f8ae0d7c,
            ]))],
            [Felt::new(BigInteger256([
                0x58567561e7c75a12,
                0xa1738456b107e2c7,
                0x0a97f047062cfd49,
                0x050111f3fa1befb6,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_4_3::compress_k(input, 4));
        }
    }
}
