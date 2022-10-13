//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, STATE_WIDTH};
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

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
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

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x257511079cb0ce1e,
                0x0bb28486e5f12c4f,
                0x0eb12568400b3647,
                0x003cfc482438e1f0,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x6c50a85cdd6138c7,
                    0x956185ccd9019717,
                    0xbb3b5f31f2bbe138,
                    0x11b5b77f0fdbcd3b,
                ])),
                Felt::new(BigInteger256([
                    0xcfdcb0300cb2abed,
                    0x09ef6252fadd2a9c,
                    0x8e8c166360189840,
                    0x0563634e8f16638c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3b9faedc5aba1c7a,
                    0x5f1aa5b2cab5d169,
                    0x20a0b2ae58592447,
                    0x095ea460f2e9d9f0,
                ])),
                Felt::new(BigInteger256([
                    0xd794da6a5048cf4f,
                    0x4f6ddfadb1d31fa1,
                    0x151f360e534675d3,
                    0x0fa24cd450734fcc,
                ])),
                Felt::new(BigInteger256([
                    0x892eeb472637c79e,
                    0x30a29dffaf25749c,
                    0x42cedef0a431df0e,
                    0x060738c1f2a839fa,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x303eadf8ebb93a0e,
                    0x62ebb7d002aeab10,
                    0xc0c422e425364cab,
                    0x050e7b2b8c8c9738,
                ])),
                Felt::new(BigInteger256([
                    0xfa652a3c1e8312f8,
                    0x6fdc1a060a464729,
                    0xef2d6fa79505af09,
                    0x03ca8c3ce47a6d61,
                ])),
                Felt::new(BigInteger256([
                    0x952a8f364f5ddb29,
                    0x863aae287360ac92,
                    0xf31163c39c0d0b9f,
                    0x09034f3db60a499d,
                ])),
                Felt::new(BigInteger256([
                    0xf501dec16671b258,
                    0x751eba693449849d,
                    0xa072d580dd80f88f,
                    0x06be41629d3929d3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbc226447f8101a48,
                    0xc98c6ebc4bbce124,
                    0x524979651d0749e9,
                    0x092cc6b5ca6f57c2,
                ])),
                Felt::new(BigInteger256([
                    0x1e94ebf266b6c914,
                    0x2e056a4c8093a61f,
                    0x68c93cd33e36a22a,
                    0x00596626f0965fc5,
                ])),
                Felt::new(BigInteger256([
                    0x7ea21671cf42a6a2,
                    0x48eff65b9c1ab5a3,
                    0xe5aa1d673a064c70,
                    0x00803e6c3a7a60b8,
                ])),
                Felt::new(BigInteger256([
                    0x029cd37ee61b2b20,
                    0x9ee6f54fa04072bb,
                    0xc729faea35a45b26,
                    0x083e5913da0b6adb,
                ])),
                Felt::new(BigInteger256([
                    0x876d5c3f1d35eb42,
                    0x7e1427e6660d09a9,
                    0xfe806c3f5ddb8a53,
                    0x07be46319146140a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x347ee685fac03d0a,
                    0x9c9d5017b04a2a05,
                    0xa89fd838e86a84ac,
                    0x0aa4338da84cd001,
                ])),
                Felt::new(BigInteger256([
                    0xc86fa1080f1e1941,
                    0x80380f94cc066ee2,
                    0xd2e6234b2b5571bf,
                    0x08161ddd10d83bf3,
                ])),
                Felt::new(BigInteger256([
                    0xa0d8ffca28dc47aa,
                    0x88b4dd430cdc93f3,
                    0xfed4cffa59e26936,
                    0x0720c180b228aace,
                ])),
                Felt::new(BigInteger256([
                    0x19619ae4f37913b5,
                    0x6d6e9a896f1e0ede,
                    0xa6a2a4bacde7130d,
                    0x0bf4b529cc712288,
                ])),
                Felt::new(BigInteger256([
                    0x27c877ba619f06bf,
                    0x48ab665567d37123,
                    0x94b70a8d492ed819,
                    0x0a32b561ba54eb2c,
                ])),
                Felt::new(BigInteger256([
                    0x3125cd36fd3070d4,
                    0x4391ab18cce9898b,
                    0x6d6272bd9b01f802,
                    0x064acf2835d39770,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xdafc038069fd5db5,
                0xeac3bdcb4dc4c176,
                0x327c7f03aaa6c9e2,
                0x1038ef41eb071bce,
            ]))],
            [Felt::new(BigInteger256([
                0xa0cc5a1057849a16,
                0xc021449cbf9f1f9a,
                0x41a32b4c19848976,
                0x11ce73d35759c503,
            ]))],
            [Felt::new(BigInteger256([
                0xa3cde68c386bef14,
                0xf2b7d2d01e87770b,
                0x64d51e261f6a9886,
                0x0c13669ce7a8fe7c,
            ]))],
            [Felt::new(BigInteger256([
                0x3b95dc32e7c73ca0,
                0xec23385108f18fa9,
                0xbbeed92ecca0abba,
                0x0779307a894b58d5,
            ]))],
            [Felt::new(BigInteger256([
                0x8f56e8871e66321c,
                0x1816eb430d503f5e,
                0x847da650bedf281c,
                0x069fd124748a9e4f,
            ]))],
            [Felt::new(BigInteger256([
                0xaf8455b26e3f95a6,
                0x392b960594e13f7d,
                0xbe06082e5ff25134,
                0x0487ec00c6a16afd,
            ]))],
            [Felt::new(BigInteger256([
                0xadf8582982d7df7c,
                0x62fa9b63b494aa5b,
                0x49d4bbae5366eb71,
                0x07c048ad4083ad98,
            ]))],
            [Felt::new(BigInteger256([
                0x9e8e55860d928cfc,
                0x8cce5061e179bddc,
                0x2a3b529327be2c18,
                0x0bcdf69b9fcd5401,
            ]))],
            [Felt::new(BigInteger256([
                0x1813e91c9d53686f,
                0x63b75eccf91b7e3b,
                0x2a60a117bfd58db2,
                0x04605e12e71b3901,
            ]))],
            [Felt::new(BigInteger256([
                0x5be52378a79ebc5e,
                0x7c6e8bdc15c2e20f,
                0x57fe62e1544f6a6d,
                0x08e07629951fd305,
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
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x1a8af368e6127347,
                    0x4fe2733f740d9aac,
                    0x2a54351dfc9cb887,
                    0x0e456cf53b178598,
                ])),
                Felt::new(BigInteger256([
                    0x33c4409222a3e554,
                    0x69b6d28b539b72c6,
                    0x008e3e9b0d654a8a,
                    0x125c5f24085e8c1c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9b6405991ab1067a,
                    0x87aa5b283cad9a87,
                    0x940d5bcd560a967a,
                    0x0708974d68bb1b3f,
                ])),
                Felt::new(BigInteger256([
                    0x1d0fa598be1ff33f,
                    0x8f55f467c83cf2d7,
                    0x3622c8d5633961d0,
                    0x0cd714e5fc6b41b2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x531a197f2518d4fb,
                    0x8aaa9f0bdafe7b5a,
                    0xfd66b0744400b2ca,
                    0x11353bd579213180,
                ])),
                Felt::new(BigInteger256([
                    0xbce70bc0249c0d0e,
                    0x0103e8fa85e526bb,
                    0x147ba6cb5d57bf94,
                    0x0399ad8bb5e15b8f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc8364ec14a17aad8,
                    0xa6c8c6fc1ca58e5d,
                    0xc961e42895daf1a2,
                    0x0533068671593d1b,
                ])),
                Felt::new(BigInteger256([
                    0xebed402b7a2ea5eb,
                    0xed44e4728d723d03,
                    0x91511c21b61a0e03,
                    0x02105c68b4c7d641,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xea7004f50bcc9674,
                    0x2bd29b75404bd8ac,
                    0xa280880eda1751e4,
                    0x0a5fde9167c28372,
                ])),
                Felt::new(BigInteger256([
                    0xb0b7781090ce1502,
                    0x4d2255caf9c7f85f,
                    0x9e6dd678dad6e180,
                    0x0f0492b911f15087,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x29bd303259301abb,
                    0x026665452d76f71f,
                    0x639630e4571f8434,
                    0x087f6206e7c40a7b,
                ])),
                Felt::new(BigInteger256([
                    0x4bf2328db2f61456,
                    0x72441dd2ba0e6f0b,
                    0xd79a070392746754,
                    0x022447d9692ec440,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xc25c681464ef9efc,
                0xb33164d52901298d,
                0x47638f376748f63a,
                0x0c854c445d88ab3e,
            ]))],
            [Felt::new(BigInteger256([
                0xc1f4dc66f6f142b0,
                0x2d53c171c4864654,
                0x7a862779707cf03d,
                0x0d2a593155f3946f,
            ]))],
            [Felt::new(BigInteger256([
                0x7c31bd3d9062263e,
                0x8a2ea43cad6d7ceb,
                0xbcc51af78e687007,
                0x0d1ee9f3d147ebef,
            ]))],
            [Felt::new(BigInteger256([
                0x98e67f66ef5f7c0b,
                0x538042664ed54e9f,
                0x6adde7dd05addd9a,
                0x0d429fa6cb27775d,
            ]))],
            [Felt::new(BigInteger256([
                0xfa33e0f7237a4166,
                0xb335dfe41e1062a9,
                0x2b107dc3ce2091e0,
                0x0517f5aa7131285c,
            ]))],
            [Felt::new(BigInteger256([
                0xf261aa37093dcae6,
                0xbfc753a9d893cd9e,
                0xafab184470afe7ea,
                0x07197d743b215138,
            ]))],
            [Felt::new(BigInteger256([
                0x8111bd8c1575cdb8,
                0x4aa09b0b674ef29d,
                0xc0d1473be5211268,
                0x1278e1f30f183537,
            ]))],
            [Felt::new(BigInteger256([
                0xa704ae1046c8ce58,
                0x34a7769a71da7dba,
                0x246ca5429f0a9fb1,
                0x084d99f6f2989b6d,
            ]))],
            [Felt::new(BigInteger256([
                0xdb7e662473a62f9b,
                0x3843b70ce26b6c04,
                0xccbde5e83e15bf55,
                0x013c34783cddbe99,
            ]))],
            [Felt::new(BigInteger256([
                0x26d39ef86ba76ac6,
                0x9175aede1752734c,
                0x90c004421c44eb43,
                0x00f440cc6daf2924,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }
    }
}
