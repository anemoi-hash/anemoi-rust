//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiEdOnBls12_377_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiEdOnBls12_377_4_3 {
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
                AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
            AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
                AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
            AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
        AnemoiEdOnBls12_377_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiEdOnBls12_377_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiEdOnBls12_377_4_3::permutation(&mut state);

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
        AnemoiEdOnBls12_377_4_3::permutation(&mut state);

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
                0xe2b879eac8a4fa2b,
                0xd688dac0a1ee3afa,
                0x98c62e76ba6550ab,
                0x0dcfb25d3bd51da5,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xc09fbfe566e3b57c,
                    0x7593e3d7da2471fd,
                    0x7a7bc47491155fba,
                    0x055aff2ad9ef1470,
                ])),
                Felt::new(BigInteger256([
                    0xe624961f6975054a,
                    0x9a9e40e1a6c1ac38,
                    0x72fd09dc0e698f7f,
                    0x0712f3534d8d388e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x23f141f484c57755,
                    0xcef3b72d66d13713,
                    0xf5b2b8ee6e02d3b8,
                    0x05086ff2ded34b44,
                ])),
                Felt::new(BigInteger256([
                    0xeb7c1a05d7963336,
                    0xf016bce4dc662ce8,
                    0xbfc67102ef8f3d76,
                    0x0754f99757b56d3e,
                ])),
                Felt::new(BigInteger256([
                    0x76eaa553502717ef,
                    0x2a2544ccb77552e1,
                    0xf789ed899d87899c,
                    0x069982cb562f376e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1452a594257ab066,
                    0x68f0abd4b5f70cff,
                    0x27900d67bfed6719,
                    0x11e0bf02d0fa0c8e,
                ])),
                Felt::new(BigInteger256([
                    0x57a111724d60d518,
                    0x876cb010adf0f821,
                    0xead43bccfd75c00a,
                    0x096ca91022aea233,
                ])),
                Felt::new(BigInteger256([
                    0x030816b723fbd622,
                    0x48f2cae48760762d,
                    0xf185d709dce3cd4e,
                    0x03d1624752077f89,
                ])),
                Felt::new(BigInteger256([
                    0x9d881e5f1bde68be,
                    0xf086f44daeac562a,
                    0xdb0ead75576f300e,
                    0x021149ebb8fc9747,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaf9be7a490845e01,
                    0xdd2da42461f53aa7,
                    0xc11a255825345253,
                    0x03c7d4f62f419345,
                ])),
                Felt::new(BigInteger256([
                    0x56e91b7144a3a07f,
                    0x9ad7ccbebc0c5d13,
                    0x13bef88fbd8d9e74,
                    0x05a94469b97e7f28,
                ])),
                Felt::new(BigInteger256([
                    0x17d9eef59e0acfed,
                    0xfc61419369137c85,
                    0x7c42eb143d5dcef0,
                    0x06f67ead7c6ed6b2,
                ])),
                Felt::new(BigInteger256([
                    0xe61e1a94753b0b09,
                    0x712f081ab3c92e95,
                    0x803d6c64c414a62d,
                    0x0e5b7ff74ae099c6,
                ])),
                Felt::new(BigInteger256([
                    0x0397f8fbf53e7c25,
                    0xbd2556282f36b004,
                    0xed97bbcdb16e1641,
                    0x0fd9020129597645,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7a16d8c94831ee07,
                    0x086a865f705d8385,
                    0x23447fbd736cff16,
                    0x003df863618e0f4b,
                ])),
                Felt::new(BigInteger256([
                    0x1a58ec27590908d0,
                    0x88288c229fa3f43b,
                    0x947abf09c6da9fc6,
                    0x0382aa464a29b16f,
                ])),
                Felt::new(BigInteger256([
                    0x03e2b2f2fcbb0a57,
                    0x637426de8600a67b,
                    0xbbdea6c32d7ad87f,
                    0x11258ba44487ca93,
                ])),
                Felt::new(BigInteger256([
                    0x61f15767be42763d,
                    0xcd7f8136388a54ab,
                    0xf43d605382eb924c,
                    0x0093a5c1a6e45533,
                ])),
                Felt::new(BigInteger256([
                    0xe687218bcc700420,
                    0xe98cdb1fa1a7061d,
                    0x36cb16fde5eadff5,
                    0x0fd10b0e1c2660f8,
                ])),
                Felt::new(BigInteger256([
                    0x4a330a826ac47852,
                    0x3a8b19440a51917d,
                    0x9ad02e7133e8e382,
                    0x04572f920eca0378,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x2617ff8213eb6bd2,
                0x02dae5c01fcfda1c,
                0x97cefcfe09f198d3,
                0x05c36a7b54c0afec,
            ]))],
            [Felt::new(BigInteger256([
                0x004fb9370593c267,
                0x34022d2a9023898b,
                0x4024ed2e2331e301,
                0x0dbf3aeffdbfe708,
            ]))],
            [Felt::new(BigInteger256([
                0x8f43eea0087ae93d,
                0x928d9a343f4ef9ba,
                0xd3f7e4329099719f,
                0x0a0eb82c98c08931,
            ]))],
            [Felt::new(BigInteger256([
                0xb10276ec6a3c5976,
                0x95af3656b32980b7,
                0x66b04ec6ec427c5f,
                0x0118919d08a208b2,
            ]))],
            [Felt::new(BigInteger256([
                0x771cb62c724d4b5f,
                0x27d9059d05a21fc1,
                0x8200df25574e1b2c,
                0x0cdd5bb537d41cbb,
            ]))],
            [Felt::new(BigInteger256([
                0x92a83e545291477a,
                0x6a7159e6664e1533,
                0xaada644bc25b2654,
                0x002074f50da0b0ce,
            ]))],
            [Felt::new(BigInteger256([
                0xf28a4998a77845cc,
                0x15d6753d77d817d6,
                0xb790508409c2637e,
                0x05d87e9568d98885,
            ]))],
            [Felt::new(BigInteger256([
                0x0d80efd864558cd4,
                0xc2c6a34e8cb5eced,
                0x8389b3d6af00cc9a,
                0x1211a2ae1fbd8cba,
            ]))],
            [Felt::new(BigInteger256([
                0x72f218ef8967c8e5,
                0xc99e817aa168dc49,
                0xd9421d13c96bac9d,
                0x074224acc2aa322a,
            ]))],
            [Felt::new(BigInteger256([
                0x36755a5327d6d7db,
                0xd5bfd7899f4ddb34,
                0x091883990e4ccfbc,
                0x085c5e70b78df05e,
            ]))],
        ];

        for (index, (input, expected)) in input_data.iter().zip(output_data).enumerate() {
            println!("{:?}", index);
            assert_eq!(
                expected,
                AnemoiEdOnBls12_377_4_3::hash_field(input).to_elements()
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
            [Felt::new(BigInteger256([
                0x2617ff8213eb6bd2,
                0x02dae5c01fcfda1c,
                0x97cefcfe09f198d3,
                0x05c36a7b54c0afec,
            ]))],
            [Felt::new(BigInteger256([
                0x004fb9370593c267,
                0x34022d2a9023898b,
                0x4024ed2e2331e301,
                0x0dbf3aeffdbfe708,
            ]))],
            [Felt::new(BigInteger256([
                0x8f43eea0087ae93d,
                0x928d9a343f4ef9ba,
                0xd3f7e4329099719f,
                0x0a0eb82c98c08931,
            ]))],
            [Felt::new(BigInteger256([
                0xb10276ec6a3c5976,
                0x95af3656b32980b7,
                0x66b04ec6ec427c5f,
                0x0118919d08a208b2,
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
            assert_eq!(
                expected,
                AnemoiEdOnBls12_377_4_3::hash(&bytes).to_elements()
            );
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
                    0x137fb2096b6372fc,
                    0x993312bcb39674f3,
                    0x0ee2c095c075446e,
                    0x07bcb037d09845d8,
                ])),
                Felt::new(BigInteger256([
                    0x9c94353b8cf22b89,
                    0xedca14f946bd9ae1,
                    0xe918e80a6cb5548c,
                    0x0a658aa3f63b81a1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5d672ce69e30a2c7,
                    0xa0c96ddc675f8ccf,
                    0xb0e30b8e3ab3a2d1,
                    0x0efd6b5cbfad6f33,
                ])),
                Felt::new(BigInteger256([
                    0x7139e5927f5d485b,
                    0xbcc379ea4ebd513a,
                    0x08f4f21a365bdd85,
                    0x117e4efb37d6c61c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9ba2832ea60c63e5,
                    0xc618da490250d3c6,
                    0xbb1d5f3c3f801348,
                    0x00cf318fca099907,
                ])),
                Felt::new(BigInteger256([
                    0x15ced790b9fb251d,
                    0x735c35bffde94d19,
                    0xf5f7bc2bb6a0b785,
                    0x10833ad21b37f94c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7dfd898d7d0d71f4,
                    0xa052d5dbd5ab9d86,
                    0xc7138911383ee30c,
                    0x0280ecb99159af2e,
                ])),
                Felt::new(BigInteger256([
                    0xf58c8da2e78ed2b4,
                    0x9cfa2a78fd84181d,
                    0xc268625a37bff577,
                    0x059ad657e2fc355f,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiEdOnBls12_377_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected.to_vec(),
                AnemoiEdOnBls12_377_4_3::compress_k(input, 2)
            );
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xb013e744f8559e85,
                0x86fd27b5fa540fd4,
                0xf7fba8a02d2a98fb,
                0x12223adbc6d3c779,
            ]))],
            [Felt::new(BigInteger256([
                0xc48f92791d8deb21,
                0x03e270c7e61cde08,
                0x5923b08a14d7d056,
                0x0dd054f95d578ff9,
            ]))],
            [Felt::new(BigInteger256([
                0xb1715abf60078902,
                0x39751009003a20df,
                0xb1151b67f620cace,
                0x11526c61e5419254,
            ]))],
            [Felt::new(BigInteger256([
                0x738a1730649c44a8,
                0x3d4d0054d32fb5a4,
                0x897beb6b6ffed884,
                0x081bc3117455e48e,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected.to_vec(),
                AnemoiEdOnBls12_377_4_3::compress_k(input, 4)
            );
        }
    }
}
