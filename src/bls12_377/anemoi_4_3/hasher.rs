//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBls12_377_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiBls12_377_4_3 {
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
                AnemoiBls12_377_4_3::permutation(&mut state);
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
            AnemoiBls12_377_4_3::permutation(&mut state);
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
                AnemoiBls12_377_4_3::permutation(&mut state);
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
            AnemoiBls12_377_4_3::permutation(&mut state);
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
        AnemoiBls12_377_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiBls12_377_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiBls12_377_4_3::permutation(&mut state);

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
        AnemoiBls12_377_4_3::permutation(&mut state);

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
                0x45af572b53b5e957,
                0x4656a1d6d9fc0590,
                0x7103dbd2f03b6211,
                0xe483926a3f17a0dd,
                0x140d8b92b7f2348e,
                0x00713db7c8c863ca,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x88d4f300b6ffaba5,
                    0x32756436576e43ad,
                    0x83b32ac7aad0c063,
                    0xf624c26b1b5c6a3c,
                    0x6fa658b4109ebb5a,
                    0x00ab806ac64eb078,
                ])),
                Felt::new(BigInteger384([
                    0xf1472555b2f40f32,
                    0xaeb89e2a74efc646,
                    0x9c14d17e4b8aeb4f,
                    0xc5e258087e42eb0b,
                    0x334d322a6415eb27,
                    0x005cdbd8e138b878,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x8f8fedfdcbef1a7e,
                    0x31b8d53226868867,
                    0xafa1410ffe0f8d94,
                    0xa1b4960d10fbaf1d,
                    0x1e65d4874fc114fc,
                    0x01aa132dccd517d8,
                ])),
                Felt::new(BigInteger384([
                    0x8a8b03413351509a,
                    0x7b954efaebae33d4,
                    0xc6439e517900195b,
                    0xe1c812152c1fa65f,
                    0xa6b19d50eb7138c3,
                    0x01047af60cf5f20d,
                ])),
                Felt::new(BigInteger384([
                    0x671c2ea6338c5a02,
                    0xafbb8b927e524f8a,
                    0xb53b72c614b29886,
                    0x033f1511e0dcad92,
                    0x24a2a09409168516,
                    0x000528a667a89817,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x9b72adeff6b6f564,
                    0x7abd71883d56b3a4,
                    0xde984b8975efcdce,
                    0x735985060b466507,
                    0x0d2b6523c5c072dd,
                    0x006553dc3c447c72,
                ])),
                Felt::new(BigInteger384([
                    0x9bb942d362ea236d,
                    0x85e598576295cbd9,
                    0x40da6c6981334438,
                    0x7c7a89c444fb473f,
                    0xbc753deefbcdcf81,
                    0x00960cb5d4a557b0,
                ])),
                Felt::new(BigInteger384([
                    0x53c17863ff65d4b0,
                    0x1942297151e4741e,
                    0xc3d125c02fd8b5d1,
                    0xd1b847c0047c42d8,
                    0x1d9285bcd99e3e72,
                    0x00fe094aa188cce9,
                ])),
                Felt::new(BigInteger384([
                    0x35bd8871b39a85c0,
                    0x8a3ef67e5d281b80,
                    0x3721df41d5aefd80,
                    0xd53d82f669cb6448,
                    0x19402e4321659b3c,
                    0x018bf79b4888f01f,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe122a0170c782c15,
                    0x4ab6745aff75819d,
                    0xe85cd91484c9c50a,
                    0x1ea21c1664b4d786,
                    0xf585da0a67654a06,
                    0x0063b261308040cd,
                ])),
                Felt::new(BigInteger384([
                    0xdc4d20f0eaf1e4eb,
                    0xa3fd91a3e3de3706,
                    0x448be03caa2ef5f1,
                    0xb2148d2992a45491,
                    0x2cd4ca97aa47bbee,
                    0x01a0cd2203899fc9,
                ])),
                Felt::new(BigInteger384([
                    0xfc99a78d0d4d8ebe,
                    0xa52e36e9e103947f,
                    0x7103fdf6623c3c8b,
                    0x3cbd8aca7f1a83fa,
                    0x44741f6c2bcdce9f,
                    0x012e90bd974ad521,
                ])),
                Felt::new(BigInteger384([
                    0xfd313c5e3026b40a,
                    0x60c50eab42523180,
                    0x1cc276e43d5e915b,
                    0x65c663e3bf882405,
                    0xad57d6fd46edb194,
                    0x0198ee1b33ae161d,
                ])),
                Felt::new(BigInteger384([
                    0x1c550b7c988b3d72,
                    0x3e0380f6263ae5e5,
                    0x81853c059825bb08,
                    0x74ad4cdf2229e5ab,
                    0xa04a166175a4835e,
                    0x00570369f717a7f4,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x9e6068a9eea03fba,
                    0xe60758528683b55c,
                    0x67508832d5af1167,
                    0x309d502fc7750b02,
                    0xff07cf2054060df4,
                    0x01477c369a64ffb0,
                ])),
                Felt::new(BigInteger384([
                    0x9b80325851db0847,
                    0xf48ee04055dc1c5d,
                    0x5ffff20b5679c9c3,
                    0x79c8a22157104dc1,
                    0x74d337a4ad6b273d,
                    0x017b039e1a3a5d03,
                ])),
                Felt::new(BigInteger384([
                    0x75464f1088559312,
                    0xe08b39badfe705c5,
                    0x0c0fa9b37717d235,
                    0xe10e293860ecd898,
                    0xce8331e6d9eb5fa1,
                    0x00223e8442761020,
                ])),
                Felt::new(BigInteger384([
                    0xb7bd0ad20377271c,
                    0xb6e69cd95c94caaf,
                    0x9c728044240d0abf,
                    0xbc6e68c913551ce6,
                    0x504e07f8ae9f2f03,
                    0x01107e9438fbbe44,
                ])),
                Felt::new(BigInteger384([
                    0x6595727d4637433a,
                    0xb623efa53e0c841d,
                    0x407d30af52e22fa6,
                    0x26bc60c1471290c2,
                    0x47cf5517233891ab,
                    0x01a874e2490d3ba1,
                ])),
                Felt::new(BigInteger384([
                    0x29fd5b2022fe6664,
                    0x1483d3ccbd9a02a2,
                    0xd2da306cddc65e60,
                    0x57c2cf5fed1f9c22,
                    0xd5ee7770e1e13f34,
                    0x0007edcae6a69560,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xb38930e8a62394c7,
                0xed299fc6440370ec,
                0x145a80d27247542f,
                0x69a66da685501b16,
                0xd391b946a65a6e94,
                0x0097557d512b92d0,
            ]))],
            [Felt::new(BigInteger384([
                0x5e4ec18c4e8c877c,
                0xed5abc82acf45abe,
                0xefb483594da0360b,
                0x1a2fe40426aaf4a2,
                0x52c28f3cd5762efc,
                0x019b6987b21cc770,
            ]))],
            [Felt::new(BigInteger384([
                0x304bc423be1ab685,
                0xe6772e1232245066,
                0xf8f31e3d4ae985bd,
                0xe053574e3e34939f,
                0x5485d602e534e750,
                0x00b60a61049d40cb,
            ]))],
            [Felt::new(BigInteger384([
                0xfeec14d4be4a5b9d,
                0xa5927486be6dfee1,
                0x4a7f314927a49181,
                0x52de0da27b3dfba2,
                0xf66bd83ab3070d77,
                0x014a65318127771c,
            ]))],
            [Felt::new(BigInteger384([
                0x8492c706b6273f9d,
                0x2a98c1db63af8e82,
                0xb31c54ea7f53f4d7,
                0xca5cf29164753c8f,
                0x3f520b5e3d425213,
                0x00f02afad85237f0,
            ]))],
            [Felt::new(BigInteger384([
                0xb53d11de5269b9ba,
                0xa3b66db6194a1b6c,
                0xae97c419dcfd3fa2,
                0x05e90f7892b484bf,
                0x69c0b59381a7e075,
                0x00f7064a89a51aee,
            ]))],
            [Felt::new(BigInteger384([
                0x49dcb7c3e43dfcd6,
                0xb40b333962f052ad,
                0xaf42f8b589d4cb10,
                0x1252218371753706,
                0x66ba407aa8ccf57c,
                0x00d38ac374c4feab,
            ]))],
            [Felt::new(BigInteger384([
                0xe8529d7c2087b7c4,
                0x9eb87f4ae477e099,
                0x07e92a0d657a558e,
                0x7fa5804171a8344d,
                0xef8e080c3a614975,
                0x0059a160ef1dc931,
            ]))],
            [Felt::new(BigInteger384([
                0xbdf1a82b6d58a0da,
                0xaca8e60fab67dea5,
                0x9311196ff9b0d02a,
                0x8852fe43eddc6df3,
                0x541001453ec8e28f,
                0x0126eb45f402783a,
            ]))],
            [Felt::new(BigInteger384([
                0x46b79bd9792e754d,
                0x72fe8242e9358b5e,
                0x5ff1272305bc70ab,
                0xe817231389477468,
                0x0232a6604b05e56a,
                0x00bb6fd703b8a8ed,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_377_4_3::hash_field(input).to_elements()
            );
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
                0xb38930e8a62394c7,
                0xed299fc6440370ec,
                0x145a80d27247542f,
                0x69a66da685501b16,
                0xd391b946a65a6e94,
                0x0097557d512b92d0,
            ]))],
            [Felt::new(BigInteger384([
                0x5e4ec18c4e8c877c,
                0xed5abc82acf45abe,
                0xefb483594da0360b,
                0x1a2fe40426aaf4a2,
                0x52c28f3cd5762efc,
                0x019b6987b21cc770,
            ]))],
            [Felt::new(BigInteger384([
                0x304bc423be1ab685,
                0xe6772e1232245066,
                0xf8f31e3d4ae985bd,
                0xe053574e3e34939f,
                0x5485d602e534e750,
                0x00b60a61049d40cb,
            ]))],
            [Felt::new(BigInteger384([
                0xfeec14d4be4a5b9d,
                0xa5927486be6dfee1,
                0x4a7f314927a49181,
                0x52de0da27b3dfba2,
                0xf66bd83ab3070d77,
                0x014a65318127771c,
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

            assert_eq!(expected, AnemoiBls12_377_4_3::hash(&bytes).to_elements());
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
                    0xcee32c93af08ca0d,
                    0x11865a300381e5a7,
                    0x471676abb884aea2,
                    0xf8d9fbaf807a9ee6,
                    0x4f2eb1fc416d22f0,
                    0x012745731829c63c,
                ])),
                Felt::new(BigInteger384([
                    0x77b4fdfbb113e6af,
                    0xbf7dc79458d9b163,
                    0xf6e30c29ad0db190,
                    0xd01817652b7e0ed1,
                    0x64f8fd2ab734c4bf,
                    0x004412e33dd9c107,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x549559675f0cc21d,
                    0x616f57f2c517b005,
                    0x97ab973596b72b1f,
                    0x64682fb44939342f,
                    0x1942ba7667e6c6c3,
                    0x00b5da58b67ac6d4,
                ])),
                Felt::new(BigInteger384([
                    0x03b1617eab5c3f31,
                    0x296543a43b61f115,
                    0xee6551edf65c3ff5,
                    0x05d5149b6b5e091a,
                    0xa51d5aaef259613b,
                    0x01753281780116d2,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xcdedeea5f5049302,
                    0x07486358b161510f,
                    0x95a8f1d712f524d6,
                    0x7a6314e1ea4698d7,
                    0x51ac0611318b474a,
                    0x002b32320f5ed297,
                ])),
                Felt::new(BigInteger384([
                    0xf8f32721754e3b5f,
                    0xe97fc5a907f2758c,
                    0x0080331adbb9dc79,
                    0x6872938008402e74,
                    0x4360a163ed124924,
                    0x015d2ea037f3cc74,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2602fd1c1f67a977,
                    0x7a4b2dcde6490fd7,
                    0x69c15642b130a361,
                    0x2206fdbbc8af27dc,
                    0x03f992a2883019ce,
                    0x015361fb468d0785,
                ])),
                Felt::new(BigInteger384([
                    0xda465b65f8ac9985,
                    0xe857f20639306963,
                    0xbea40feaf9019ba4,
                    0xe316e5d77f477499,
                    0xf0b6ae96e04eaea6,
                    0x015929c5de79d7f9,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x46982a8f601cb0bc,
                0xd10421c45c5b970b,
                0x3df982d565926032,
                0xc8f21314abf8adb8,
                0xb427af26f8a1e7b0,
                0x016b585656038743,
            ]))],
            [Felt::new(BigInteger384([
                0xd33dfae60a69014d,
                0x73c93e52d079a119,
                0x671d86f3d30a2314,
                0x501a6a5cb3a229bb,
                0xf8250f64ed9edec3,
                0x007cd29416b6ccbb,
            ]))],
            [Felt::new(BigInteger384([
                0xc6e115c76a52ce61,
                0xf0c82901b953c69c,
                0x962924f1eeaf014f,
                0xe2d5a861f286c74b,
                0x950ca7751e9d906e,
                0x018860d247529f0b,
            ]))],
            [Felt::new(BigInteger384([
                0x7b409882181442fb,
                0x4b97c28fef79793a,
                0x097203fdf028f706,
                0xeafb09a0470188e7,
                0x2e753b78fbdd7f39,
                0x00fe517b0d41ce94,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_4_3::compress_k(input, 4));
        }
    }
}
