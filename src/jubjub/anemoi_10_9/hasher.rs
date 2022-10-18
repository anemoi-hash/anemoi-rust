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

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x9a48a463069c3fb3,
                0x1e9c7be4bc8b28bf,
                0x65cfe9ff9c49aec3,
                0x500dde34c405c2e4,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xb216419711721c49,
                    0xd48fc5f6d6cc0688,
                    0x7a0d03302a76b458,
                    0x1a87e779dc11208b,
                ])),
                Felt::new(BigInteger256([
                    0xaf32fa6050e18c08,
                    0x26e870ddcde4cb5e,
                    0x7a2e253ceb913087,
                    0x0469349bc816ec3c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa625e8e802275fad,
                    0xd4ddaa9b1b576f67,
                    0x01bf3383e61a39a7,
                    0x6da4183a535aff56,
                ])),
                Felt::new(BigInteger256([
                    0x6e5b511e579eeac9,
                    0x024a8cac5c70b751,
                    0x283861d780d1912a,
                    0x2aac32cd4831f996,
                ])),
                Felt::new(BigInteger256([
                    0x156f146436e6736a,
                    0xe296a535a7aa3663,
                    0x50807825ff4e6ca6,
                    0x189d11c6a0679cae,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc947824dba45c028,
                    0xe0d78582657dd7eb,
                    0xf8f542a3942fcf65,
                    0x1fa39cff470633aa,
                ])),
                Felt::new(BigInteger256([
                    0xc4062594ea614b85,
                    0x839733a750cc07e9,
                    0x5c8e422a84076c57,
                    0x717a8fbc725216d2,
                ])),
                Felt::new(BigInteger256([
                    0x603413b17c23b560,
                    0x8a373f525bd2ecdd,
                    0xe3cd9c5720baff99,
                    0x5ff26b0541b01e49,
                ])),
                Felt::new(BigInteger256([
                    0x22bc37df2246bacb,
                    0x5beb57e37de59221,
                    0x40b49851683a8a5c,
                    0x6f20f38bb0e1b96e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd42535ad4c47ccf3,
                    0xa95ad2ad18222046,
                    0xf6b0c755e5900dd7,
                    0x0c3133aa336498e2,
                ])),
                Felt::new(BigInteger256([
                    0x5f9a434148c477fd,
                    0xfd2bb88ea77e7ea5,
                    0x17c00f8cd5459c69,
                    0x5598952644a1a852,
                ])),
                Felt::new(BigInteger256([
                    0x991e47f711735460,
                    0xdfec1f2906598a0f,
                    0x962db8acd1372e73,
                    0x2d974f95e3b3af75,
                ])),
                Felt::new(BigInteger256([
                    0xef8eb9ad140a2ffe,
                    0xd9b6b569365061c6,
                    0xa917c3a0e139fa5c,
                    0x51ffc23c78c68a6a,
                ])),
                Felt::new(BigInteger256([
                    0xbccff1c2e9ecab76,
                    0xdc1e79709f01893d,
                    0x84f4301c3128b3f7,
                    0x061a7076117db593,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe6e60fedea159066,
                    0xf7bc0635bf4fea01,
                    0xd8ab175ddbac3d08,
                    0x1c1b2769ae0569b7,
                ])),
                Felt::new(BigInteger256([
                    0xa63a81bb8415bd52,
                    0xfeadcc8cdacefcf6,
                    0x5320fb97459a4963,
                    0x4c99c8b43559dec4,
                ])),
                Felt::new(BigInteger256([
                    0x592dfeb08a946ba7,
                    0x4ac16c80734a4b24,
                    0x5f7a3f64715b40f1,
                    0x357e8e19714ba4e9,
                ])),
                Felt::new(BigInteger256([
                    0x5bc35b80317ab823,
                    0x1cfec94aa93ff1f1,
                    0xb443548ad7eb3ce5,
                    0x1f493e82c4eecfda,
                ])),
                Felt::new(BigInteger256([
                    0xb1dd8834b736703e,
                    0x78216b855b3ca484,
                    0x896a9b8c60d22c55,
                    0x047a290b7f17222f,
                ])),
                Felt::new(BigInteger256([
                    0x3544e97f6752e13c,
                    0x0f2633043f609177,
                    0xb561aa491cb445ec,
                    0x08ee9daac655c52f,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x62c0ab4de1733c5d,
                0x6d1b11a55b392a35,
                0xa2c2ad207dc07f13,
                0x437aa28e3cc9c8cd,
            ]))],
            [Felt::new(BigInteger256([
                0x0c10145d3b56d70c,
                0xba2c4716b6efd71f,
                0x21757e29fe5f6e62,
                0x715b9761ba0f5e3e,
            ]))],
            [Felt::new(BigInteger256([
                0xb2168e56fb40715a,
                0x91340776f5d506f6,
                0x273f089adbdc5b9e,
                0x725998633d633536,
            ]))],
            [Felt::new(BigInteger256([
                0xfd654548ea82468a,
                0x704b46de43fee389,
                0xf8e841635f3eca1a,
                0x3c03ec63034712bf,
            ]))],
            [Felt::new(BigInteger256([
                0x873f396a99f29aa5,
                0x2194e88ecfef9823,
                0x1d9c580c36665d59,
                0x440bce29dfad3a7b,
            ]))],
            [Felt::new(BigInteger256([
                0x3b28f27221c0b0fe,
                0x5b3c13c1705fedfd,
                0x8db447b87270c6be,
                0x03797d38398d531b,
            ]))],
            [Felt::new(BigInteger256([
                0xc5f872cde22e3644,
                0x375fbc11b8d77e67,
                0x3697692a03efc944,
                0x2db957d71c709606,
            ]))],
            [Felt::new(BigInteger256([
                0xbefb6e87bc3f134a,
                0xe52564004c276102,
                0x78927fbf1d7561c1,
                0x5de5d6603c982b79,
            ]))],
            [Felt::new(BigInteger256([
                0x4184d8068fc2197a,
                0xc950dd87608c7ab0,
                0x2f5825abb3699f1d,
                0x442c186a60674207,
            ]))],
            [Felt::new(BigInteger256([
                0x3286f4fd515a9171,
                0x93003d7111c11374,
                0x3f917437d8bf8724,
                0x6503b2bcf89cd7dc,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf37260cdd6752080,
                    0xfc1f3991e7d97ec3,
                    0x78e26095aa9db840,
                    0x5738cd9598cb10d5,
                ])),
                Felt::new(BigInteger256([
                    0x00e25645148be0ea,
                    0x935165c38271e05b,
                    0x1af2d289c3272f12,
                    0x0c9c613b427db16f,
                ])),
                Felt::new(BigInteger256([
                    0xda84bd4404bb0ea0,
                    0xf07b04e35a54dbfd,
                    0xfedda77bfdea8239,
                    0x2a12c9e83f9196b6,
                ])),
                Felt::new(BigInteger256([
                    0x1200ce42f0833a3e,
                    0xfb83618b56141617,
                    0xddff97e416a3fa5a,
                    0x4ad37e0513e7f2f5,
                ])),
                Felt::new(BigInteger256([
                    0xf395180e8368385c,
                    0xc71d9c038e6e5ef0,
                    0x3964d723ebd1ca60,
                    0x04381b6d07d8773c,
                ])),
                Felt::new(BigInteger256([
                    0xbdea6fa6862ef3c4,
                    0xb52686376f166134,
                    0xd9095a346bd737e2,
                    0x2262c319304e7657,
                ])),
                Felt::new(BigInteger256([
                    0x5def83ebb66ef010,
                    0x6bec043098835efb,
                    0xf4eb01033e44285b,
                    0x4452c71a8a879521,
                ])),
                Felt::new(BigInteger256([
                    0x030c2a00b60df2f4,
                    0x81dfd0b018031832,
                    0xd4a304aa5918ab13,
                    0x2518f41686115b02,
                ])),
                Felt::new(BigInteger256([
                    0xb5f51106a864a980,
                    0x3cb7159ff5dab137,
                    0x636b7106311e03de,
                    0x1c35498656bf5a0d,
                ])),
                Felt::new(BigInteger256([
                    0x0a6e645318747e26,
                    0x16015a51939469b7,
                    0xd422d0e35d4b674d,
                    0x368941b8d0b84b90,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x33feafeff615c773,
                    0xee732a33f42c00f2,
                    0x5cb26f2c2360168f,
                    0x6495a932d31b0359,
                ])),
                Felt::new(BigInteger256([
                    0x6f93677c27c37d89,
                    0x3b4f40369ad1750f,
                    0x62a6cb40159c9712,
                    0x46c27b234d67db4f,
                ])),
                Felt::new(BigInteger256([
                    0x49b017590eab4cc3,
                    0x60c7356efc26aa3c,
                    0x1fa4da7a46d92d0b,
                    0x22a665fac32a5d33,
                ])),
                Felt::new(BigInteger256([
                    0x343c29af038e327e,
                    0x682eb9395f52e4b8,
                    0xb15e425a2d9c32f1,
                    0x4e43136b3afbaabf,
                ])),
                Felt::new(BigInteger256([
                    0x072281575f03ff01,
                    0xbb643d0cd402fe0b,
                    0x356333ef0c8993e1,
                    0x4077b535b217dc22,
                ])),
                Felt::new(BigInteger256([
                    0xd79e61d5dee57aac,
                    0x88685ad5259b7d00,
                    0x07fd424a4fa18da0,
                    0x3aabee8294f0202d,
                ])),
                Felt::new(BigInteger256([
                    0xa5f39009cd18c7fb,
                    0x6a3c8cb3e5664332,
                    0x2f10edaa133d7cf5,
                    0x6d69bba734032f94,
                ])),
                Felt::new(BigInteger256([
                    0xfbf65c584f88ca8f,
                    0xefb3c322b6fa6832,
                    0xdc470f5f56e088d6,
                    0x4983da4dd7a3a1c6,
                ])),
                Felt::new(BigInteger256([
                    0x83bf35fd52dcebd1,
                    0x71718e707d68be21,
                    0xef35d6f1677b88ff,
                    0x1d7f9af628c07125,
                ])),
                Felt::new(BigInteger256([
                    0xb0d15eaf0e01f5df,
                    0xa255512d53db3365,
                    0xedeb5f2771a1c8cd,
                    0x4b016d22a3a991e9,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xde3d85b1d44b7100,
                    0x9dde82c9048397a3,
                    0x8f2095efc3f3fd85,
                    0x45c201a79ea45467,
                ])),
                Felt::new(BigInteger256([
                    0x7fcdf33d8937e412,
                    0xcc1a22d3dabc96de,
                    0x40bab7b33c97c2c4,
                    0x3d920fd1029639a5,
                ])),
                Felt::new(BigInteger256([
                    0x155b2a08da9dacb1,
                    0x8567e7c370a54b71,
                    0x46061e4f1547986d,
                    0x35c7f13b36e92000,
                ])),
                Felt::new(BigInteger256([
                    0x7966a8bdc9067a52,
                    0xcc9e12b7a3ec02e6,
                    0xaabd59352401e55c,
                    0x0732afdee2d70d3c,
                ])),
                Felt::new(BigInteger256([
                    0x8e65ec1ed66b96dc,
                    0x938758169a581fa8,
                    0x3643793ea05659cf,
                    0x3c0edfea4174193a,
                ])),
                Felt::new(BigInteger256([
                    0x0b668f7552319511,
                    0xfa9d55837f01a008,
                    0xf3dd820985f25431,
                    0x1c0866c0caaded83,
                ])),
                Felt::new(BigInteger256([
                    0x9fecb836396a4013,
                    0xf3ec7080c2f4bd85,
                    0x8598f8ac754d5c25,
                    0x1bdc508b04de7386,
                ])),
                Felt::new(BigInteger256([
                    0x31a9536da372fa94,
                    0x255dfb7f07b40502,
                    0x6b9da6ae25d415b0,
                    0x5c1e62fca153e4f8,
                ])),
                Felt::new(BigInteger256([
                    0xfd542bae1f3f475c,
                    0x41d65c715d72afd1,
                    0xf2d741bfe397408a,
                    0x4952bc102264659d,
                ])),
                Felt::new(BigInteger256([
                    0xe7dbac46bab9bec0,
                    0x531080a9229692dc,
                    0xd6302c040a7132e7,
                    0x597706ff6ded4336,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5e6537675d9e6d8f,
                    0x89e8adc0080f88df,
                    0x95261d678ac362bb,
                    0x1950979df937d4e0,
                ])),
                Felt::new(BigInteger256([
                    0xc9c886a15ad1af29,
                    0x6a8100c9cb3e9172,
                    0x289c8e8ea2a4d9d4,
                    0x068191120e5523a8,
                ])),
                Felt::new(BigInteger256([
                    0x1c5528771af50c6b,
                    0x6382f7a8d3cc8cac,
                    0x8591d50ff2c50bae,
                    0x5b73b250b0a453e1,
                ])),
                Felt::new(BigInteger256([
                    0xcfb91ec046c24d18,
                    0xcb3d50ed14eedb41,
                    0xd710c089e1bdcdaf,
                    0x29dc3634f738483d,
                ])),
                Felt::new(BigInteger256([
                    0xedb1aee06cc2dae8,
                    0xba866175772b7bac,
                    0xb873cff7e9d95b4c,
                    0x4eb73483e43bcf93,
                ])),
                Felt::new(BigInteger256([
                    0xd5e41a77b86fc26d,
                    0xa1cae597006df72e,
                    0x29a83f0eb8780ec0,
                    0x3300d725c9899baf,
                ])),
                Felt::new(BigInteger256([
                    0xd3e6710f57a6ae38,
                    0x62f22f56d7d81e1d,
                    0x4bd661ed70b74f31,
                    0x492fbf3632fb0432,
                ])),
                Felt::new(BigInteger256([
                    0x20e76c173609a6be,
                    0x928ed6224c6db90e,
                    0xb31735bf9ca78faa,
                    0x5e6ae23f11461eb9,
                ])),
                Felt::new(BigInteger256([
                    0x3e5698db12a36c3c,
                    0xe4bb781ddd5d7868,
                    0x555b424830f00a90,
                    0x39a09225cc9e32f2,
                ])),
                Felt::new(BigInteger256([
                    0xd2557066f1f47127,
                    0xafc8ccb5642ae79b,
                    0x511fe56216cbd37b,
                    0x1db3e01fec39c8d4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfe1568528a085793,
                    0x61b8d954fd30a075,
                    0xb455b7584dc83be2,
                    0x0f07b0eaec5f502a,
                ])),
                Felt::new(BigInteger256([
                    0x24b5a0f161c6e55d,
                    0xfc2fa8821b3ff616,
                    0x77ff54de1893b62d,
                    0x35835d9e40a727a3,
                ])),
                Felt::new(BigInteger256([
                    0x9be6f3283bc10e28,
                    0xdb26c0c82a0ee0fe,
                    0x6e1c9fa36a7fff94,
                    0x097577053475b516,
                ])),
                Felt::new(BigInteger256([
                    0xeb4578f413e5380a,
                    0x4b73fbceedd02e90,
                    0x2491f130ed302044,
                    0x069ac719202bbab6,
                ])),
                Felt::new(BigInteger256([
                    0x2f1f40aec59afc97,
                    0xbcb87cdca57434c1,
                    0x5a8c8c5aac4db4f3,
                    0x16fa42d582c7b344,
                ])),
                Felt::new(BigInteger256([
                    0x1a23e4de9127f27a,
                    0xc90f6744cb4f0cff,
                    0x601102a940388d6b,
                    0x6b369945f4612c92,
                ])),
                Felt::new(BigInteger256([
                    0xb3bd113c9640ebc1,
                    0x17b6f7f56ac14cce,
                    0xebca5b233098c82f,
                    0x2e88cfc300eaef33,
                ])),
                Felt::new(BigInteger256([
                    0x9811ff2788514186,
                    0x33e41aee96d2dde3,
                    0xb7d23ea01ad6f4e4,
                    0x2660eb00cd89acb5,
                ])),
                Felt::new(BigInteger256([
                    0x00e1af28cc722102,
                    0xac7f40ad5bd228c6,
                    0xdd4b7bab8eddb51b,
                    0x2d91708e24fe56cb,
                ])),
                Felt::new(BigInteger256([
                    0xec882f9d186c7570,
                    0xcf9aa7c0dab8e777,
                    0x25d48e4f1fe1cef3,
                    0x0106f066a6af8609,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0644a505c694bdd1,
                    0x47b6b34c745339d0,
                    0xa3bfe542de0818f0,
                    0x4b6e8a840f2e1402,
                ])),
                Felt::new(BigInteger256([
                    0x83f22f641fbf3f5f,
                    0xc2bc7dd1d9472a20,
                    0x338f28c701a0dbf4,
                    0x0375187abd82fe2d,
                ])),
                Felt::new(BigInteger256([
                    0xbb3a72f4ef0ac8d9,
                    0xe0909cc267168f83,
                    0xe66da41bf26c68db,
                    0x544a06a3c58e7489,
                ])),
                Felt::new(BigInteger256([
                    0x77d3d776caa425b8,
                    0xd57444e837720003,
                    0x99f61988cda77d0d,
                    0x08c7ecf798a601b9,
                ])),
                Felt::new(BigInteger256([
                    0x65af381ec92268da,
                    0x4e8cc3152ae34433,
                    0xd6ef533008e332b4,
                    0x5df790c88b6253b4,
                ])),
                Felt::new(BigInteger256([
                    0x16edb5f74c9e70ad,
                    0xe35438a7a8538430,
                    0x67454041f6133a5b,
                    0x5f0fe5659a08530a,
                ])),
                Felt::new(BigInteger256([
                    0xbba6d35687d9d05b,
                    0x054442b8fa4f5568,
                    0x315d26c1a2f67ef6,
                    0x2cb621a84572c9f1,
                ])),
                Felt::new(BigInteger256([
                    0x32e870c3ef5e748a,
                    0xc93254bd70384427,
                    0x079a7071baf37f71,
                    0x0fdbf7fc5598595c,
                ])),
                Felt::new(BigInteger256([
                    0xbef286ceb9f0ddf0,
                    0x77a76d09d2ebd81b,
                    0x65e3f2f493258be6,
                    0x472718c999c84516,
                ])),
                Felt::new(BigInteger256([
                    0xae48cbd249c9541b,
                    0x3d20dcd118efecd1,
                    0x881c92270eb080c5,
                    0x66ca8ee37d616b2e,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x55ff008803acfb27,
                    0x57c700d6a4e83dc2,
                    0x17243956565ef726,
                    0x3c0b330e239755eb,
                ])),
                Felt::new(BigInteger256([
                    0x17274ccef06a21b2,
                    0x072faa33d878897a,
                    0x72778d22bd74de34,
                    0x42a2b5cac0531ecd,
                ])),
                Felt::new(BigInteger256([
                    0xf98fa6508deaeada,
                    0x0d54137c793a9642,
                    0xcb4daa8fc01f8224,
                    0x4d633a8dedee6343,
                ])),
                Felt::new(BigInteger256([
                    0x1ebcddf0402cd3d9,
                    0x2870ac7089c0b367,
                    0x6f56348429614b23,
                    0x12a23ecf4ef26f07,
                ])),
                Felt::new(BigInteger256([
                    0xfb00f10ea8c25de7,
                    0x379b3086eee19275,
                    0x48c43f5378e8b776,
                    0x66270582b7a52d82,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x437ad9a2b5e471d2,
                    0x865e691e7aa731ed,
                    0xcd20276c686d9f45,
                    0x55e8a79fea4775a9,
                ])),
                Felt::new(BigInteger256([
                    0x1141c15384d41fb0,
                    0x5a363f3e5021ed22,
                    0xe490824276718dbb,
                    0x31d776c13b82f406,
                ])),
                Felt::new(BigInteger256([
                    0x36d4db3eaa88dbf4,
                    0x22231a5f3e1498f5,
                    0xe51e2a57435ad227,
                    0x0421f55566906531,
                ])),
                Felt::new(BigInteger256([
                    0xa9cdcbbbf8eb0dac,
                    0xfcd26fd45c27bcc7,
                    0xf9d1a33d199a9160,
                    0x599f816e31b72794,
                ])),
                Felt::new(BigInteger256([
                    0xd127a4c13a12424e,
                    0x82f5b8e3db9b186c,
                    0xc5f11e7d977c3590,
                    0x694e22af7ee00bae,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd84eb65a06f735e5,
                    0x45ac5f4a80cd94b0,
                    0x9e981153d37746cb,
                    0x15ac7141571ddc58,
                ])),
                Felt::new(BigInteger256([
                    0x2f4c549613fed8b8,
                    0xcb1e3b55da1b1f33,
                    0xecefdb8168d9287b,
                    0x09e25f67f8fcf136,
                ])),
                Felt::new(BigInteger256([
                    0x9a690cb0879e3787,
                    0x5c8efee81a6cca46,
                    0xd72d70c09fef14db,
                    0x16bbd7d6659044d8,
                ])),
                Felt::new(BigInteger256([
                    0x55fbec8861734bd4,
                    0x4ad4ae811338411d,
                    0x9fd67e92c3d3d1c9,
                    0x1accaea65592e56a,
                ])),
                Felt::new(BigInteger256([
                    0x683b3a3ae7fadb40,
                    0xf45932f042db3b05,
                    0xba01012f8a480a53,
                    0x4939592c3ef43b70,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa35601aa0cfa3398,
                    0x067bc7e26718e33b,
                    0x2e415ba2cccd5bc5,
                    0x0877691ad78f95da,
                ])),
                Felt::new(BigInteger256([
                    0xeccf878add4819e6,
                    0x3b70a9a1f590cb25,
                    0xe01c4332171f2d10,
                    0x179b000ee93f4a87,
                ])),
                Felt::new(BigInteger256([
                    0x5597be87de92ea7a,
                    0xf2ebccabbe933249,
                    0x85d487a3d1f51a89,
                    0x1d3e30444c6af90f,
                ])),
                Felt::new(BigInteger256([
                    0x6629a3fb0b5521c0,
                    0x458cce8166429f84,
                    0xd6d02832bf23f340,
                    0x20a8a942769b44e0,
                ])),
                Felt::new(BigInteger256([
                    0x30900543d27d112c,
                    0xd61772515462fbbc,
                    0xd7f4c629f572f44f,
                    0x36fc92d2a278e28f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xac541d9b2fecf183,
                    0x253286e494c0ae13,
                    0x890b864fa1476a98,
                    0x03d28ef38f5bfda6,
                ])),
                Felt::new(BigInteger256([
                    0x697d8f8577a83890,
                    0xed6056a2ad1fb91c,
                    0x472b60a9ceaba68d,
                    0x3d6141d8933f25b3,
                ])),
                Felt::new(BigInteger256([
                    0xfaefd5a55aa1f098,
                    0x08e3515e67009e52,
                    0xea37ac0cb7dccf96,
                    0x2d4411ce587539a9,
                ])),
                Felt::new(BigInteger256([
                    0x33775c32daffebde,
                    0x824fca865ed7edb8,
                    0xd8c02bbf7514e637,
                    0x6083ac0f2d8b68e2,
                ])),
                Felt::new(BigInteger256([
                    0x9e2d27f82b4ce3b8,
                    0xc493b5ecba7e3681,
                    0x9b5915cb012a6d65,
                    0x22e4e48fe348b160,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb4eb9f7397998dfa,
                    0x982903bbd4d2280d,
                    0x886f2edc1599407b,
                    0x4da70fb88ec3cd2b,
                ])),
                Felt::new(BigInteger256([
                    0x7d4c07c31368cec6,
                    0x844ca2e3295c25ac,
                    0x6b4cf5f011e1dabf,
                    0x72001e64be47d755,
                ])),
                Felt::new(BigInteger256([
                    0x1e0a3016094dff65,
                    0x7dc607b22726238e,
                    0x6aeb7d820a6f9b9c,
                    0x09da8b9bf4378dc7,
                ])),
                Felt::new(BigInteger256([
                    0xf6538b01ded456b1,
                    0xb8010c46f72e5002,
                    0x0d362050b67b5c89,
                    0x3a1a50d256108225,
                ])),
                Felt::new(BigInteger256([
                    0x4628be5d86c1be18,
                    0x584c013e4946a2a5,
                    0xac68ae0e4a30e77c,
                    0x549e1006809395f0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4e7d88f77327b253,
                    0x28024ffa22bb5464,
                    0xc24ed0ee418af190,
                    0x529f788dd3de5fa0,
                ])),
                Felt::new(BigInteger256([
                    0x38ad50dbf9af0564,
                    0xc5d3571bbe22bd9c,
                    0xc3915bbbb6a217b2,
                    0x56bcc211b788270e,
                ])),
                Felt::new(BigInteger256([
                    0x0a3154160ad36eec,
                    0x0219187bcfe2fd5f,
                    0x886883ed2b5d2be5,
                    0x70a4ebd133f4bf07,
                ])),
                Felt::new(BigInteger256([
                    0x018379abe3c0f070,
                    0x2e9dbd7606c73329,
                    0xce6c4b35475de8b9,
                    0x1f45e9494963c5a2,
                ])),
                Felt::new(BigInteger256([
                    0x8efc55df24622672,
                    0xb4292ece18de74b8,
                    0x2424fb3182815e3d,
                    0x2046b6e0e87b3127,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfdb2b1a3780337c6,
                    0x9053c0a3d8070193,
                    0xb0ec5440f6efa987,
                    0x0bcb9a9ecef601af,
                ])),
                Felt::new(BigInteger256([
                    0xb287084136983373,
                    0x0557d8117af64b02,
                    0x4edba5964e5be9f3,
                    0x62b618338e9dcc1f,
                ])),
                Felt::new(BigInteger256([
                    0x202a586b9f317cb0,
                    0xc10bb27e8648b913,
                    0x41d53d4e6f53cad7,
                    0x4b8d512a8424074a,
                ])),
                Felt::new(BigInteger256([
                    0x988bce64cecb36ff,
                    0xdc6f9501b088482d,
                    0x53b238a97aed2bc8,
                    0x28dbabfdc099f665,
                ])),
                Felt::new(BigInteger256([
                    0x5b9796523dc5cdcc,
                    0x9942c585014c053e,
                    0x17855f12ad1b03fb,
                    0x4cd336b0b77dcf72,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5b39b914f5906d18,
                    0xc1b99fd4cd52b45c,
                    0x17ac0ac6a08ad6ac,
                    0x731a67ac8d3c27e5,
                ])),
                Felt::new(BigInteger256([
                    0xe8d27aa8727c678e,
                    0x3a02cc0675f7b3bc,
                    0x2b065d2fb7bef534,
                    0x2abc0ada99575c69,
                ])),
                Felt::new(BigInteger256([
                    0xae5c12b0acdf77e5,
                    0xdbb1713f5b1420ec,
                    0x7eede0f679f5a4d2,
                    0x1050acf28c8c7949,
                ])),
                Felt::new(BigInteger256([
                    0x786f097b2cdf3003,
                    0x1ff97361c7d54eb0,
                    0x1540a931e95d1964,
                    0x67fb53b00284793c,
                ])),
                Felt::new(BigInteger256([
                    0xe930f89585b67aa1,
                    0x4416142421efc174,
                    0x6523770cccd05c62,
                    0x5beea12b8a7da434,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5fb1dc0b39bcde78,
                    0xd1a21a77020f223b,
                    0x33de37888b6314cd,
                    0x0c0e235526537f3b,
                ])),
                Felt::new(BigInteger256([
                    0x163334526a25c025,
                    0xf75c94bac688326c,
                    0x672bd350c4705374,
                    0x066ebb122b443b45,
                ])),
                Felt::new(BigInteger256([
                    0xd127b65e2070f609,
                    0xfab96d531bc5c7ce,
                    0x1d37b4a4e651a1dd,
                    0x0fb87e637b4ee4af,
                ])),
                Felt::new(BigInteger256([
                    0xc9c45cf42acf28f3,
                    0xae062a0ed2981234,
                    0x2651f9a6a9017168,
                    0x18fc5b9bfca1e6d4,
                ])),
                Felt::new(BigInteger256([
                    0x9ada009bcd2c6830,
                    0xf2301e5887f49141,
                    0xedd2c58a22547e8e,
                    0x4b33553b33ab77d6,
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
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x28ec05f9860b561e,
                    0x6bf7370ab793c55f,
                    0xd3120d25f5762431,
                    0x550abcb2059c0cbf,
                ])),
                Felt::new(BigInteger256([
                    0x2f1964ced54f6ac8,
                    0x472b39e5d671fade,
                    0x27bffbb210e85e3d,
                    0x22aa6ea110f64c71,
                ])),
                Felt::new(BigInteger256([
                    0xbf0fef2cb6d72925,
                    0xd74e1175358fadc1,
                    0x7c38b024670e58ea,
                    0x24d7824fcf3b4056,
                ])),
                Felt::new(BigInteger256([
                    0x0e3cc57505bde583,
                    0x864e5b4c1c77b684,
                    0xaf850169651ba89b,
                    0x238328d9c35aafe1,
                ])),
                Felt::new(BigInteger256([
                    0x859847662b96a94f,
                    0x00f13d0421e43e97,
                    0x45e7d05b20657407,
                    0x54bb1e9dc761ef27,
                ])),
                Felt::new(BigInteger256([
                    0x42c810da5be8cf8a,
                    0xb4d26c2d39a962e7,
                    0xbf079eccf03ab18c,
                    0x45fe646e13d0435d,
                ])),
                Felt::new(BigInteger256([
                    0xa669861f9029a7c9,
                    0xf4f590e4958e50a3,
                    0x0d887199b74b2cba,
                    0x0c3b5182aea32b0c,
                ])),
                Felt::new(BigInteger256([
                    0x7fbf132558108563,
                    0x1fce989342ad6618,
                    0xa2102b7aa8b3470e,
                    0x171a15d2c9c4bc86,
                ])),
                Felt::new(BigInteger256([
                    0x9820507e1f0ef091,
                    0x26d15293dc483ce4,
                    0x161e21b1eed1ab4b,
                    0x31e1ef0979b47487,
                ])),
                Felt::new(BigInteger256([
                    0x7c80ec3f55bc285d,
                    0x6100a290440a6746,
                    0xf21015b9f0c1e4b6,
                    0x022571f7a5abce8e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x07c72a12919ab6c2,
                    0xea8b0bc754f4ee33,
                    0x3e93a8e95f1d9e41,
                    0x0144f36ca33ca860,
                ])),
                Felt::new(BigInteger256([
                    0xf698ff3c30a1a24b,
                    0x198989b980e6ce2b,
                    0x1276536c47625de0,
                    0x4fbcfc4bdbaf2d0b,
                ])),
                Felt::new(BigInteger256([
                    0xc6549b6b37ca264c,
                    0x8c0243f2cfccf955,
                    0x8e58e0674ba12c4f,
                    0x4328d28c8c659ec7,
                ])),
                Felt::new(BigInteger256([
                    0x64a0ce5adde0807e,
                    0x3880091cf2623262,
                    0xf9528f7d3c6bac30,
                    0x73724d58504f95a9,
                ])),
                Felt::new(BigInteger256([
                    0x118da55c2fb09a5f,
                    0x46aaa1e754dd1764,
                    0x5019764bac971dbd,
                    0x490a8179c8fc4e33,
                ])),
                Felt::new(BigInteger256([
                    0x819e42a3b1e426de,
                    0xf5acdbf346fb8c91,
                    0x943e93eec227dfe7,
                    0x1f5b177b4c35f8cc,
                ])),
                Felt::new(BigInteger256([
                    0xe82ef092feae36aa,
                    0x1943e0686c4fd008,
                    0x75b0f37fad264a5c,
                    0x13d3c79f93c928f7,
                ])),
                Felt::new(BigInteger256([
                    0xe981e1038b09ba1d,
                    0x673f293e4ee668a8,
                    0x8ab7667a27980c89,
                    0x142ba84d80ff5739,
                ])),
                Felt::new(BigInteger256([
                    0x59b4371cf962f656,
                    0x07b551957f88f91d,
                    0x15d81b9c83383e56,
                    0x717cd033a375e847,
                ])),
                Felt::new(BigInteger256([
                    0x1b488f83316b6606,
                    0xed253c6e81721592,
                    0xb7a41822a61972f1,
                    0x27cec44ce36392ca,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xec2fbeeab92b0362,
                    0xf4609acd04ebea30,
                    0x1f8b910ac5184d31,
                    0x07dead3801b28a1a,
                ])),
                Felt::new(BigInteger256([
                    0x3d714eddb0defdfe,
                    0xcc3fb6fdbd526103,
                    0x9af6bbdf51ad1ed1,
                    0x4104b2e9b3d2b24b,
                ])),
                Felt::new(BigInteger256([
                    0xf25e2faa4bf7cc65,
                    0x08ab396750607719,
                    0x580f81459850890e,
                    0x6c0c8ef0ac8079f2,
                ])),
                Felt::new(BigInteger256([
                    0x2f4cd76316c887fd,
                    0x8d5d6c3637168ac0,
                    0x716fbf64bcecb0be,
                    0x4a82f70ab96f6433,
                ])),
                Felt::new(BigInteger256([
                    0x7f7a42c65b1d9265,
                    0xf7a58f9f92197424,
                    0x30d76d0c79a6c6c3,
                    0x1ee31344c7bff20a,
                ])),
                Felt::new(BigInteger256([
                    0xc3135a7822dbd05a,
                    0xf742a07e8498b059,
                    0x198d71b474e09b80,
                    0x0ded80f8743627f7,
                ])),
                Felt::new(BigInteger256([
                    0x668e01a6e5462421,
                    0xff2d9c4ae1d29fe7,
                    0x7c11b76f881460ad,
                    0x4b4a3c62e3992859,
                ])),
                Felt::new(BigInteger256([
                    0x924cec19c3ee0e1d,
                    0x197342ed79d0cd1f,
                    0xd9d42f1c1627fad4,
                    0x69bc3c61bcc71bba,
                ])),
                Felt::new(BigInteger256([
                    0x44c0d83e3b8cb18a,
                    0xf47b597cd7b8238b,
                    0xbe4d9e6b8069c738,
                    0x4d8f0df47b7f2331,
                ])),
                Felt::new(BigInteger256([
                    0x2c64e3acdabb0e99,
                    0xac5062d9ddfc996d,
                    0x31c7d66393076505,
                    0x525c88af9c2b3374,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xddfc7116ad55f32e,
                    0x05b8ed5046a775c3,
                    0x987d0f22918c845f,
                    0x0879f199927d7d9e,
                ])),
                Felt::new(BigInteger256([
                    0x6d86888a72d7312b,
                    0x62552800e145e806,
                    0xd2b2e6f7ff448dec,
                    0x31f641211dea9668,
                ])),
                Felt::new(BigInteger256([
                    0x5f9aa6b24fcb1e5b,
                    0xd1e2c1371eb035cb,
                    0xc73d33dad5541487,
                    0x3441750fdc6c7893,
                ])),
                Felt::new(BigInteger256([
                    0x342e5a319d657321,
                    0x3a9678ee071fe512,
                    0x207bb38cdd317358,
                    0x56287ba6a6c13275,
                ])),
                Felt::new(BigInteger256([
                    0x7d26f2a85b540380,
                    0x0bbe46fa722f9f3c,
                    0xbc32b805851de71c,
                    0x6f39c218e55fd3bc,
                ])),
                Felt::new(BigInteger256([
                    0x070f6bf381948c76,
                    0x1365a47d19419663,
                    0x74db506ef018d73d,
                    0x6a9a98fc9164bc4a,
                ])),
                Felt::new(BigInteger256([
                    0xf79ba1138f1742cf,
                    0x30ccc7420c371b86,
                    0xdb9b798b327f51b5,
                    0x54fad41e3d0eb76b,
                ])),
                Felt::new(BigInteger256([
                    0x91ddfba84321e2e3,
                    0xb1d0dbf404b711f9,
                    0xf6cb4be81e8b7b38,
                    0x32e41c9a7241f5fc,
                ])),
                Felt::new(BigInteger256([
                    0x53672f4721ef6f91,
                    0x7eaaff556d29d59a,
                    0xe39f747ad1c2e162,
                    0x543563e7da710f1e,
                ])),
                Felt::new(BigInteger256([
                    0x776492b353c617e3,
                    0xe49d06ef90233ead,
                    0xb553af74f3297750,
                    0x1fc252242f946532,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xda17e97e29f4715a,
                    0x5058af71fd7dedab,
                    0x5d8797aba4d5123b,
                    0x7041627a3e2937f6,
                ])),
                Felt::new(BigInteger256([
                    0x0c8c1c6cbf72c1be,
                    0x05c4729dcf4ac4d1,
                    0x78ecba597cbdf37d,
                    0x19f91f355a9e854e,
                ])),
                Felt::new(BigInteger256([
                    0xf0da05356d245431,
                    0x6e29e1570979f051,
                    0x190c1cd0285ffd06,
                    0x00e7668f45027d85,
                ])),
                Felt::new(BigInteger256([
                    0x3e0e15ac4394a0f7,
                    0x59cf17ec05ab47af,
                    0x87fd39509944a9ac,
                    0x3c5379b969849f40,
                ])),
                Felt::new(BigInteger256([
                    0xc71a0da6048910f2,
                    0x5bfa1c21d93f6ae3,
                    0x11411cdaa945ec23,
                    0x1d1da0c1a0c016fd,
                ])),
                Felt::new(BigInteger256([
                    0x4c4e816c5bf35691,
                    0x2dd837e3eccb5cfd,
                    0x33332e0fda6e7693,
                    0x6b969a7612a7ef18,
                ])),
                Felt::new(BigInteger256([
                    0x6c9969848297bb5b,
                    0x24ef41d105cfa1a4,
                    0xb072e7d83c762ed2,
                    0x3260fcac19c71638,
                ])),
                Felt::new(BigInteger256([
                    0x93e9bf96697bba75,
                    0xd49702c4d7f30864,
                    0xb66713425328b297,
                    0x6ed0d94b93ca4c60,
                ])),
                Felt::new(BigInteger256([
                    0x21011aebed803223,
                    0xeca57c2f025f0061,
                    0x965f42d307d36272,
                    0x27d36091f024c648,
                ])),
                Felt::new(BigInteger256([
                    0xfd301f2690826425,
                    0x359866f1af2396e3,
                    0xb518e2095ae59ab3,
                    0x1a3d781b0b17e065,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x77e9bf283f951f82,
                    0xa7dcffd4e168550f,
                    0xf32b1bb5b215e119,
                    0x5f9ffb29f762356b,
                ])),
                Felt::new(BigInteger256([
                    0x745f89477724f4db,
                    0x6e155d62c3286935,
                    0xa2b65e924916310c,
                    0x4420f387ac45d7c8,
                ])),
                Felt::new(BigInteger256([
                    0x49f28c6c7274b63d,
                    0xbf60b4a151a033d3,
                    0xe98b3b9d5a7e7338,
                    0x737e16a32504fb89,
                ])),
                Felt::new(BigInteger256([
                    0xc0647a1fcdfa4ec2,
                    0x2267d0f045f58a2e,
                    0x19006a433199abec,
                    0x6bdc739d51194008,
                ])),
                Felt::new(BigInteger256([
                    0x086d9f6f6c1a7743,
                    0xe444510cfe723675,
                    0x80f4365095c9e9dc,
                    0x6e0d5575df220432,
                ])),
                Felt::new(BigInteger256([
                    0x94397ee6cfcdb195,
                    0x03248f28f68047a8,
                    0xe6bb489e2581bd79,
                    0x6d22ecbb18e6496a,
                ])),
                Felt::new(BigInteger256([
                    0x9c0eb8941ce6f9ef,
                    0xd27c3767197d5a58,
                    0x527fe34b6742fab1,
                    0x0cf4b284be42fce1,
                ])),
                Felt::new(BigInteger256([
                    0x226d6015398efc47,
                    0x7e6186e32107d1e6,
                    0x587a938d1d8e9092,
                    0x5b7a3f97bb21c7e5,
                ])),
                Felt::new(BigInteger256([
                    0x9ddc4deed1b22856,
                    0xe72ddca55e0d0f8a,
                    0xf1872b345ff20853,
                    0x420aba3498dcd650,
                ])),
                Felt::new(BigInteger256([
                    0x7c131f77c258c39e,
                    0x34c50b7fd55dd252,
                    0x0047a7642ae83c37,
                    0x0cbb1d57f29cf078,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x8073c2a86af13971,
                0x24db53786f40eb5e,
                0xa69034d062f9aa0d,
                0x5cff1912853579f5,
            ]))],
            [Felt::new(BigInteger256([
                0x0686e6b4183ebd6e,
                0xdb04a36e40a3d53b,
                0xf01de5b0c00d160e,
                0x66f4692de9b70795,
            ]))],
            [Felt::new(BigInteger256([
                0x603b3e64ec026d37,
                0x58c9d6f6cb6a9e4e,
                0x8953055020b9883a,
                0x266308ff2094b5fb,
            ]))],
            [Felt::new(BigInteger256([
                0x7c76f0fca6a76ae3,
                0xfcbedaffd5e41fec,
                0x0fbd3ccd60d6b2e9,
                0x21082e2ffcb0839a,
            ]))],
            [Felt::new(BigInteger256([
                0xa5646b06faf3b449,
                0x78b4a21c869972c9,
                0x119b741049323f3e,
                0x41c9404f37b8dd90,
            ]))],
            [Felt::new(BigInteger256([
                0x21f7b06f3543aab5,
                0x45d4b610d33bf230,
                0x4eed27c4ecd1635c,
                0x3831aa0115056198,
            ]))],
            [Felt::new(BigInteger256([
                0x1ee643af0a13d910,
                0xf93c74cd0ddadd3e,
                0x7acb94cb7344244b,
                0x193b997d7aa3acbe,
            ]))],
            [Felt::new(BigInteger256([
                0xee94d4390af9304d,
                0x050cd8c8b358ca64,
                0x55eb89c37f56aee4,
                0x3b9f691bdad4bed6,
            ]))],
            [Felt::new(BigInteger256([
                0x3c042db49789ddfa,
                0xffbc9266e492a0c9,
                0x461b97630fc78d61,
                0x42287e89e3f350d5,
            ]))],
            [Felt::new(BigInteger256([
                0xfe6f0e5b6423ab92,
                0x2066d3d44ad0ada7,
                0x2a8900592667d279,
                0x6804aa56cc505948,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 10));
        }
    }
}
