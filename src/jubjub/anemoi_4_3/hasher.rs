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
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x70dd6bac4617e704,
                0x68bf7bef62f81fa8,
                0x91ff0b51d412986b,
                0x146acc7b785b6cec,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xef0dfe21766bd430,
                    0x51f284390a144984,
                    0x6b67365055f576b8,
                    0x49742d3318603dc0,
                ])),
                Felt::new(BigInteger256([
                    0xf7761686d6984d42,
                    0x5ff0916efb35ba4f,
                    0x43183afd0a41d04d,
                    0x410b264b14d057c1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd14c7f8111697655,
                    0xe0cedd4bd588f9a8,
                    0x1bc4ac57043a53b9,
                    0x41e99d149390cb1d,
                ])),
                Felt::new(BigInteger256([
                    0x7fa1beeac351b6c1,
                    0x4e2f22ca833ca116,
                    0xc711715421354558,
                    0x2028f9721c2114b3,
                ])),
                Felt::new(BigInteger256([
                    0x04b1d8902654c8d6,
                    0xd147b0d5ce151f5c,
                    0xea56fc4ba7096db1,
                    0x6966ac388bbf80ae,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x689c2ccca55976b9,
                    0x0f5ba69bd4f9fd5c,
                    0x29998fec0df6cedf,
                    0x476f2f28a97f1208,
                ])),
                Felt::new(BigInteger256([
                    0x5895fdc8eeb4052f,
                    0x36ea0fb450e944b5,
                    0x6e1681c2b27f7df2,
                    0x2216258f1cade51f,
                ])),
                Felt::new(BigInteger256([
                    0x0202e3a3a8fb7d73,
                    0x41bd39faa6f69b2a,
                    0x8b02d10c178dda83,
                    0x71750747fb62220a,
                ])),
                Felt::new(BigInteger256([
                    0xf407ab10cfa6eab5,
                    0x2bc71d8274e4d831,
                    0xf37a9b7fa43432fa,
                    0x73a7b3fe1ca80d57,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0c0899e1e283a991,
                    0x3892c922ac61b8f9,
                    0x1195ced8443359d8,
                    0x4ab8459636cdcef1,
                ])),
                Felt::new(BigInteger256([
                    0x794690312d136d4c,
                    0xccb6ab1691aabd0d,
                    0x5734a9453b7a6520,
                    0x17c077557c40fd41,
                ])),
                Felt::new(BigInteger256([
                    0x82eb9a4d2af9cedd,
                    0x87ca8872ae86c960,
                    0xe9a9b191ca502d8d,
                    0x356c44ce5a52ebe8,
                ])),
                Felt::new(BigInteger256([
                    0x1c1a8df02ed019f6,
                    0x5c6b195897ff4dc9,
                    0x498e52a8f096d14e,
                    0x437eb7edbf0f910d,
                ])),
                Felt::new(BigInteger256([
                    0x635e31bef7a03c2f,
                    0x393b9fbfa8c6fd29,
                    0x445d02fb52cfb375,
                    0x1cc940a4149d520e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x92a0a0f211d3b84b,
                    0xce9caf8610c9cdd7,
                    0x2193cdf0bba24728,
                    0x304b7f011fd6d3ac,
                ])),
                Felt::new(BigInteger256([
                    0x4301b13cea580692,
                    0x3af5c3da60ea7d24,
                    0xb82edcd336db6d8d,
                    0x303ae98f85357bfc,
                ])),
                Felt::new(BigInteger256([
                    0x8debecc63327d999,
                    0x96de870ea75caad4,
                    0x19b1b2db7233f3ba,
                    0x69bbb47f99397556,
                ])),
                Felt::new(BigInteger256([
                    0xd6990af4912ebf9e,
                    0x70e046959c2b2e2b,
                    0x5e6dfd699aa1bf99,
                    0x0bb6c28bc27f2483,
                ])),
                Felt::new(BigInteger256([
                    0x0bb61b9f04985426,
                    0x550ee67b5d081489,
                    0xda73f03c2e0ec265,
                    0x48ef3691e8877fcc,
                ])),
                Felt::new(BigInteger256([
                    0x165ed060568c89b7,
                    0xbbf34691cd9b2ab3,
                    0x3f24636326112d03,
                    0x6fcc64dff3a9fd05,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x918f2fa86b4497ae,
                0x599bc57573364f55,
                0x0b9b725655bf6617,
                0x546c41a408f77be9,
            ]))],
            [Felt::new(BigInteger256([
                0x02bba48a0cbfd054,
                0xcd3e9c263208bac0,
                0x21d8b10381e6fd61,
                0x4085eafbcc996b08,
            ]))],
            [Felt::new(BigInteger256([
                0x5c17550ab803e4a5,
                0xc8f1f1af1718830d,
                0x964dbe6580c0f081,
                0x6ff7dc10d7e0dbaa,
            ]))],
            [Felt::new(BigInteger256([
                0x33a7b76fa700bfb1,
                0x923831dc37302ae6,
                0x0df1514144518d8c,
                0x1d553cba55b1d924,
            ]))],
            [Felt::new(BigInteger256([
                0x8e90336dc0e99702,
                0x687e369dbdd0eec8,
                0x791d8ad2aa75129d,
                0x4f569b8464b925ee,
            ]))],
            [Felt::new(BigInteger256([
                0x987404bbc059220c,
                0x50bb9661e39c7bf7,
                0x53c6ee4e2f12f395,
                0x57abaa994eab5419,
            ]))],
            [Felt::new(BigInteger256([
                0x17c8abf191db4754,
                0x522024481bd73449,
                0xc743fc2aa8d7d739,
                0x0ef0b924528bfb3d,
            ]))],
            [Felt::new(BigInteger256([
                0x56768b6ab9fc2f9f,
                0x223bfc4fd3aeff89,
                0xb38c33d654d3b5e2,
                0x10160f3b36e4d961,
            ]))],
            [Felt::new(BigInteger256([
                0xf93c5199d8e79d6a,
                0x58cc4683127f05db,
                0xd91559bf56e95f64,
                0x290b8b5ba924c073,
            ]))],
            [Felt::new(BigInteger256([
                0xf90c976de9333928,
                0x3a78d45c78010140,
                0xa2d71af70611c0d9,
                0x361d90e97fc5ea17,
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
            [Felt::new(BigInteger256([
                0x918f2fa86b4497ae,
                0x599bc57573364f55,
                0x0b9b725655bf6617,
                0x546c41a408f77be9,
            ]))],
            [Felt::new(BigInteger256([
                0x02bba48a0cbfd054,
                0xcd3e9c263208bac0,
                0x21d8b10381e6fd61,
                0x4085eafbcc996b08,
            ]))],
            [Felt::new(BigInteger256([
                0x5c17550ab803e4a5,
                0xc8f1f1af1718830d,
                0x964dbe6580c0f081,
                0x6ff7dc10d7e0dbaa,
            ]))],
            [Felt::new(BigInteger256([
                0x33a7b76fa700bfb1,
                0x923831dc37302ae6,
                0x0df1514144518d8c,
                0x1d553cba55b1d924,
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
                Felt::new(BigInteger256([
                    0x7a62affd5acc7136,
                    0x1cd9e3a31afd8bd9,
                    0xf174f6cfdd3cb6de,
                    0x7294c9093a0edbf5,
                ])),
                Felt::new(BigInteger256([
                    0x5a69b07102e2d68d,
                    0xd3361ee8e1b34859,
                    0x39efe717c7cd0314,
                    0x0be8ca003747ea39,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4b4086a0c8662f83,
                    0xbde3599214926884,
                    0x6a149e78bffbbc07,
                    0x1d4e6def6d3653fc,
                ])),
                Felt::new(BigInteger256([
                    0xdd681f506295fe9e,
                    0x97f9f92071f2f029,
                    0x1004fde0b8274d60,
                    0x600af06c30da28a7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6798a2429c9c505d,
                    0xffbe498121b6819f,
                    0x82a9d659157828c0,
                    0x1700850cc6a442fc,
                ])),
                Felt::new(BigInteger256([
                    0x9b6a0ff06c7c8c52,
                    0x98922009352ad805,
                    0xc3746dbb6fb48f76,
                    0x2071153b7dca57c2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa51d6be6dfd2c1f5,
                    0xb64a16b71aae3234,
                    0x441dd541132bd9c7,
                    0x17abfdd4a6782b36,
                ])),
                Felt::new(BigInteger256([
                    0x79e8eea5b3789055,
                    0x196ec4e46d60f813,
                    0xc1a761a718431878,
                    0x73b32a875654b504,
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
            [Felt::new(BigInteger256([
                0xd4cc606f5daf47c2,
                0x9c525e88fcb27833,
                0xf82b05df9b67e1ed,
                0x0a8febb647b948e6,
            ]))],
            [Felt::new(BigInteger256([
                0x28a8a5f22afc2e20,
                0x021faeaf8686fcaf,
                0x46dfc4516e813163,
                0x096bb7087472ff5b,
            ]))],
            [Felt::new(BigInteger256([
                0x0302b2330918dcaf,
                0x9850698a56e159a5,
                0x461e4414852cb837,
                0x37719a48446e9abf,
            ]))],
            [Felt::new(BigInteger256([
                0x1f065a8d934b5249,
                0x7bfb37988810ce49,
                0xd28b5ee021cd1a3a,
                0x17718108d32f62f2,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
