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
        // Generated from https://github.com/vesselinux/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 12],
            vec![Felt::one(); 12],
            vec![
                Felt::zero(),
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
                Felt::one(),
            ],
            vec![
                Felt::one(),
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
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0xe3f0c9f057360c0a,
                0x3d0fd75332cb79f4,
                0xd06032643d11d051,
                0x03ea43c57d35d36d,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x65a1be0fc1e93818,
                    0x765ef1b1d6f756b6,
                    0x8ebceebca8e3d896,
                    0x4d8b5510486fff3f,
                ])),
                Felt::new(BigInteger256([
                    0xcd14abf51dc5b07b,
                    0xc10c4977c5295690,
                    0x73994ed3a6b763c3,
                    0x3c8a9db3f385c867,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc38db1a3f8d91ba2,
                    0x86b9eae59353f375,
                    0x4f04c903aad505ad,
                    0x48714b2729fc50db,
                ])),
                Felt::new(BigInteger256([
                    0x214f88de5d81c14e,
                    0x82f15adf278e5aa5,
                    0x0d093311045b58a6,
                    0x1ea3a0188eced50d,
                ])),
                Felt::new(BigInteger256([
                    0xcb1d5496f10caf0a,
                    0x86592c8c70015166,
                    0x47285c74ef9c1729,
                    0x2add80bc11a82057,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7cd084c6f5bf2647,
                    0x1c15dbecf3fd4208,
                    0x36cc64cb0cb7da1b,
                    0x09a9cfcb53263ebd,
                ])),
                Felt::new(BigInteger256([
                    0x908295e54031b1f1,
                    0xd39964f2b2a86d6c,
                    0xc72fc4126593551b,
                    0x28a1c3b396c25839,
                ])),
                Felt::new(BigInteger256([
                    0x72eae7a72ff821e7,
                    0x694687f1f23503d5,
                    0xe7490d9eeea34e6c,
                    0x0bc4ad7e968bc32f,
                ])),
                Felt::new(BigInteger256([
                    0xda9b3dff80eed1b9,
                    0xda6b4189e088745c,
                    0xd195b8355dad434d,
                    0x29619eed77f2b8f9,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xae08efc9b8cad924,
                    0xd14f7a0901449f19,
                    0xe014f672bf699cb3,
                    0x64925df4baeeadfd,
                ])),
                Felt::new(BigInteger256([
                    0xf0138db833d614a4,
                    0x14574ed1b270f577,
                    0x94239e9a51e208fe,
                    0x6daef4b2f5467030,
                ])),
                Felt::new(BigInteger256([
                    0xe6ad9e74d82e5e28,
                    0xfd4b55c882970322,
                    0x038985276fe2bde2,
                    0x5d994272d3abd0cf,
                ])),
                Felt::new(BigInteger256([
                    0x6af70d9ce244f34e,
                    0xdb1e6cb148a14f7d,
                    0x5f6f32f48b13e9ef,
                    0x293d05ecf3813a05,
                ])),
                Felt::new(BigInteger256([
                    0xe7d31b92d067b5c6,
                    0x6f9e3f1ddd2a3f9d,
                    0x4fef1c10a1c38e35,
                    0x33217d58a71725cd,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x718f2bc573b87fef,
                    0x25fef3f99799bb74,
                    0x066196585856655a,
                    0x10fd19aa6cfc4435,
                ])),
                Felt::new(BigInteger256([
                    0x464ee52093fa4f23,
                    0x7d884bfecf363450,
                    0x420b9d7ac9b85584,
                    0x211ad8a93764e865,
                ])),
                Felt::new(BigInteger256([
                    0xe4bdfd4fc6706648,
                    0x7793ba0f0ea89c29,
                    0xce7c512a3ce178e4,
                    0x10e3ee4562712b62,
                ])),
                Felt::new(BigInteger256([
                    0x09eb605792ba586b,
                    0xbecbb84ad5cb9b98,
                    0x8ca62dd23887748c,
                    0x59c3805708e12ff3,
                ])),
                Felt::new(BigInteger256([
                    0xea640d927a65b4cf,
                    0x9f14e21b959869bf,
                    0x4a198d3533612b95,
                    0x67e0ca2955623fd2,
                ])),
                Felt::new(BigInteger256([
                    0x76debaf9008f4d7f,
                    0x8297e4099a46d44b,
                    0x8bb51046ea241255,
                    0x61c731ab5c26fdc6,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x168c4c1df8bbeadd,
                0x1e863bf969943ef5,
                0x75c9412032a8cc8d,
                0x3dc7bc589d8565bc,
            ]))],
            [Felt::new(BigInteger256([
                0x0ced25e29a4c620c,
                0xf876e161342fd89f,
                0x1372a131918be92d,
                0x07277df90e9c259a,
            ]))],
            [Felt::new(BigInteger256([
                0x7f33cb3b7748232c,
                0xcf075220bcef6179,
                0xdf6fc572a9300a88,
                0x29aaac910938b341,
            ]))],
            [Felt::new(BigInteger256([
                0x950f8c38fe9083f8,
                0x55f61d0df72595a3,
                0xe12a9f0dfd4897b0,
                0x16de07959f2b56c1,
            ]))],
            [Felt::new(BigInteger256([
                0x4e581fd940fbd068,
                0x52545d415516018f,
                0x6303f70ce7885c14,
                0x396c5dd5775888bf,
            ]))],
            [Felt::new(BigInteger256([
                0x145f995165894e74,
                0x2bb2ea9757019155,
                0xb1726c3bde2f8fa1,
                0x5858650d813aadc9,
            ]))],
            [Felt::new(BigInteger256([
                0xb8c8b22e56e06649,
                0x62920cfa22ccc09a,
                0x71546f8cbf852062,
                0x28a10672ad73bec5,
            ]))],
            [Felt::new(BigInteger256([
                0x377a6d6a7a1b3ab4,
                0x77f4ada078cd4c0a,
                0xd497186b9f91a863,
                0x574c1645b1af2e6e,
            ]))],
            [Felt::new(BigInteger256([
                0x4c38397f68063c98,
                0x7acf51d320b6dd23,
                0x821a40ab1ce2e872,
                0x2f7546fe499e58b8,
            ]))],
            [Felt::new(BigInteger256([
                0xa01b133a908f04e5,
                0xb940c40f2d7b1650,
                0x38adb24c67827a78,
                0x2021fd075614c8d6,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/vesselinux/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 12],
            vec![Felt::one(); 12],
            vec![
                Felt::zero(),
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
                Felt::one(),
            ],
            vec![
                Felt::one(),
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
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x049ed9871978a3ff,
                    0x8de0b9dd033da475,
                    0xdac7a763131abb3b,
                    0x12224ce44651cc93,
                ])),
                Felt::new(BigInteger256([
                    0x87bac26f8847eebd,
                    0x5d01cee1e3c2eff6,
                    0xd65da37415e3ab84,
                    0x710ace7a905c5feb,
                ])),
                Felt::new(BigInteger256([
                    0x6f7fd3aa6e8f11d9,
                    0x7a634b2d6ea1af1c,
                    0x037b650b93f621cd,
                    0x202645652dd8c828,
                ])),
                Felt::new(BigInteger256([
                    0x6a96b692070f192c,
                    0x48504607e392026f,
                    0xd766fd115388e970,
                    0x42b58862d162b64e,
                ])),
                Felt::new(BigInteger256([
                    0x1adebf476e8be0f1,
                    0xb519b42777dcff3d,
                    0xe3ae7a8b764815a6,
                    0x58128db5cf2dabfd,
                ])),
                Felt::new(BigInteger256([
                    0x77801df734389ec3,
                    0x5518edfc213be9f7,
                    0xc4b638706d0f52fc,
                    0x5ab1a80df253f737,
                ])),
                Felt::new(BigInteger256([
                    0xcb446231aabf1854,
                    0x1f8b41c5cf49f6dd,
                    0x14903fd979c7dbc2,
                    0x5c49b6ff666cd92b,
                ])),
                Felt::new(BigInteger256([
                    0x99ab8e3cd187f0d7,
                    0xa7057d21a0ddc2ff,
                    0xf1fb7537f1648c19,
                    0x6890d08efa2d56d1,
                ])),
                Felt::new(BigInteger256([
                    0x0484b4adf90f40bf,
                    0x1216c25c7b49b60a,
                    0x7db5b42e034b2323,
                    0x1b90a3d6894d3ca3,
                ])),
                Felt::new(BigInteger256([
                    0x3c83c8129e47a3c3,
                    0x08d644b38d0d73d2,
                    0xd5ed80bc342e7684,
                    0x0acc8d6ec7661762,
                ])),
                Felt::new(BigInteger256([
                    0x51ecbadb9a701310,
                    0x50c0bf085839d0dc,
                    0x836f0a98387ea9e1,
                    0x720e204c4124a31a,
                ])),
                Felt::new(BigInteger256([
                    0x603af8d875eda768,
                    0xe68ae2a90977d7fd,
                    0xc37ea8c6ec7a4db1,
                    0x4bd6b94eb3bda65e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7fafa18d0a45974e,
                    0x9b989466bd4b1b4f,
                    0xa694621e1ae591ce,
                    0x39248295cb9fdfb7,
                ])),
                Felt::new(BigInteger256([
                    0xad86608a79d26cbe,
                    0xee237f1234e4b665,
                    0x858a9a9147420e80,
                    0x4cd1f007e4f54cdd,
                ])),
                Felt::new(BigInteger256([
                    0x1fb846b962eb4908,
                    0x5531a5f92f017c3c,
                    0x29d222635e66a6ac,
                    0x58ff017732367e1c,
                ])),
                Felt::new(BigInteger256([
                    0x6c685a59b8c9c8dc,
                    0x2d929df119fd469a,
                    0x3be1474e4c3ccef1,
                    0x3513fe1d04a99220,
                ])),
                Felt::new(BigInteger256([
                    0x8b448ca88ff100e8,
                    0xa2465f3e3cb0bfae,
                    0x7ba11a6612318375,
                    0x6b171ddb4bf629d3,
                ])),
                Felt::new(BigInteger256([
                    0x6c4ded120145f21d,
                    0x83c3ee7c0e1f421a,
                    0x19ed219e443ecca3,
                    0x1c657cc7fe9d113d,
                ])),
                Felt::new(BigInteger256([
                    0x33bef60094b8bb3d,
                    0x0eeea060bf2bcac0,
                    0x2fc8459b3d9721d7,
                    0x5d224409c370da1b,
                ])),
                Felt::new(BigInteger256([
                    0xdd4b275b2b273436,
                    0xffe477636fe6b63c,
                    0xb6b162257d719bfc,
                    0x288045d2cbad7d25,
                ])),
                Felt::new(BigInteger256([
                    0x25ee5fcc570ffca1,
                    0xed4bbb21dc5def3e,
                    0x351978cace27ceba,
                    0x137e5dcb7907736b,
                ])),
                Felt::new(BigInteger256([
                    0x7146d2b07184b2fa,
                    0xc86f28274eda3386,
                    0x5abcbdce49c1ad1d,
                    0x3f4b55e795c3938c,
                ])),
                Felt::new(BigInteger256([
                    0x89b03f09b4c4a012,
                    0x1ec5a39bca3b2dd6,
                    0x787eed1eb6b9cfe2,
                    0x25267b3c7ac51859,
                ])),
                Felt::new(BigInteger256([
                    0x10560b3a222b88ab,
                    0x7a1ae18cd682d8bd,
                    0xc99edc002ae2478c,
                    0x1170994f1c02c145,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdf0101379954d1ae,
                    0x7ea9b96aaed8917e,
                    0x6ddc752c3a0e4027,
                    0x63c03a1138d25ba7,
                ])),
                Felt::new(BigInteger256([
                    0x5cab2e01655e7c85,
                    0x755361c5d79faddc,
                    0xe079d84f737d1c05,
                    0x4d39c83a958435da,
                ])),
                Felt::new(BigInteger256([
                    0x81a12632279b376b,
                    0x941c3e6db4716d2f,
                    0xa0415b2b373e06a6,
                    0x05f25e8165d5c74f,
                ])),
                Felt::new(BigInteger256([
                    0xa675df4d793077ac,
                    0x483cf5fb17344bba,
                    0x13a94f91dac6ab09,
                    0x5bb075811b8e6fde,
                ])),
                Felt::new(BigInteger256([
                    0xd0a88497c90a0054,
                    0xa007e1d6eb6c667e,
                    0x3afcd93b8d61d769,
                    0x3c0fc21c679b3d1f,
                ])),
                Felt::new(BigInteger256([
                    0xc587de857865ad04,
                    0x5f86b80e8f4318b9,
                    0xbdae20735239b3ba,
                    0x30193b4c82f0bf8f,
                ])),
                Felt::new(BigInteger256([
                    0x6c5aa30a2797b98d,
                    0x6faf2ae3aeeb9076,
                    0x997f64b76fe81715,
                    0x47c649cb4986be59,
                ])),
                Felt::new(BigInteger256([
                    0xd87664ee0f7cafad,
                    0xde72d5b3ca478939,
                    0xe19b7161e12967f9,
                    0x1582ecda5a0554ff,
                ])),
                Felt::new(BigInteger256([
                    0x2cc76ae2830fdfcc,
                    0x533272a4173c95b8,
                    0xc6846bb327dfc250,
                    0x0a025ed0a466673d,
                ])),
                Felt::new(BigInteger256([
                    0x1deb2a6163b9d298,
                    0x6652732c088d3285,
                    0x4b14865ac69fbf82,
                    0x1ecc82d63f429753,
                ])),
                Felt::new(BigInteger256([
                    0x429baf37d4a1c82c,
                    0x5afb82a1592fafdd,
                    0x6fcdeb3e2ec33f94,
                    0x5eb4e9e63ab129dd,
                ])),
                Felt::new(BigInteger256([
                    0x4e9fab7de78f6bed,
                    0xa5407b0555d3634a,
                    0x517c323df93559bb,
                    0x2cff817efa2436b0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x72222c3001c0bbdd,
                    0x96f201054b3e7574,
                    0xc3a453aa2bc2d344,
                    0x3e378434236f205d,
                ])),
                Felt::new(BigInteger256([
                    0xcf477a90d334cca0,
                    0xd621efb46e982362,
                    0x380efc3d1bc4f652,
                    0x120b0eec492322af,
                ])),
                Felt::new(BigInteger256([
                    0x2ae07312f364f01a,
                    0x231055a010be1a1e,
                    0xa12659bc3259dfa8,
                    0x12883c6e155b5ee9,
                ])),
                Felt::new(BigInteger256([
                    0xd9e65d7c4c033bd4,
                    0x018db9e16b529f0d,
                    0xc897e73a8afbf919,
                    0x7055d328952ed5d5,
                ])),
                Felt::new(BigInteger256([
                    0x98405fa9855da671,
                    0xde59240e295b1d33,
                    0x9a8e505fd7844280,
                    0x003359a58f44d823,
                ])),
                Felt::new(BigInteger256([
                    0x6894f0b2187aa2f5,
                    0x40b8e4bd1359cc1c,
                    0x6df25e0f8e6bb8bf,
                    0x4dba11f60939f699,
                ])),
                Felt::new(BigInteger256([
                    0x97f0b6608fc2d803,
                    0x84f7ab9b2721a663,
                    0xb938d25d417d1be4,
                    0x2bfe6242492bf0a3,
                ])),
                Felt::new(BigInteger256([
                    0x04fb7671cc896601,
                    0x4527cd38b7fee425,
                    0x7377c44e18e2ae27,
                    0x4616c214296ef469,
                ])),
                Felt::new(BigInteger256([
                    0xf53c82839f7714a9,
                    0x326826c651165a0c,
                    0x5168b96852773fdc,
                    0x1c09a649859bd30e,
                ])),
                Felt::new(BigInteger256([
                    0x7ae2937c390cab16,
                    0x44a9d719c9f3b57f,
                    0x8b990ea17f350f64,
                    0x3c1451d09011a8cb,
                ])),
                Felt::new(BigInteger256([
                    0x77566954faa7ef73,
                    0x22ea1189394c76ef,
                    0x0015d77af99e2c0b,
                    0x2af5c6ca36fb54da,
                ])),
                Felt::new(BigInteger256([
                    0x12e4ce3910bf02f5,
                    0x4f53fdc5c927b7d6,
                    0x32f118190321c520,
                    0x44910bfe4b26de9a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcc091300ba7292a2,
                    0x5cfa9121ffce184a,
                    0x19cb7711ebefba35,
                    0x32bf3bf5a5ed15cf,
                ])),
                Felt::new(BigInteger256([
                    0x1da6f311867fe4e7,
                    0xb21e35b01890f8a3,
                    0xb58b41eca8ce3091,
                    0x09afc87f0a32a761,
                ])),
                Felt::new(BigInteger256([
                    0xf9bcb78dba3d0bae,
                    0x8bc61b6b444aa9be,
                    0x3a932ae4037f94bc,
                    0x6edf60c930355966,
                ])),
                Felt::new(BigInteger256([
                    0x6473284fbbb7366f,
                    0x3a638d40b04d420a,
                    0x5a6c9dfb56bc9653,
                    0x5f0a6a5a0ed849ba,
                ])),
                Felt::new(BigInteger256([
                    0x733bdf8158ab38fd,
                    0x23b5879eb15ba648,
                    0x617b5e1c53def357,
                    0x3b5d2f4f81f4ac12,
                ])),
                Felt::new(BigInteger256([
                    0xe70be0a65d8e6371,
                    0x3bcdfce2f808f6f5,
                    0x7eb29b6790981493,
                    0x0f6b6bd6416ad508,
                ])),
                Felt::new(BigInteger256([
                    0x2f8d544452267435,
                    0xf9e0548b1dbcb64f,
                    0x6b8a0832104e78f8,
                    0x3daf546dc32148c1,
                ])),
                Felt::new(BigInteger256([
                    0x3520eaedab731dd7,
                    0xe5bbdede88c294af,
                    0xe7e5e3278fc1f576,
                    0x19d99cae258c5132,
                ])),
                Felt::new(BigInteger256([
                    0xa6878da4d72a6c8c,
                    0xb88a5ca3a3025a9c,
                    0x685b9e58a91c586b,
                    0x561667efe84a3ebc,
                ])),
                Felt::new(BigInteger256([
                    0x31b7b1c000bacf12,
                    0xa45a8394fb82ecc4,
                    0x446a0766fcac0382,
                    0x27c15f117e7a2d37,
                ])),
                Felt::new(BigInteger256([
                    0xa0b5c3efa18dce41,
                    0xc15e7bedac05d09a,
                    0x5ee5b506b0f46342,
                    0x5ad5458eee952066,
                ])),
                Felt::new(BigInteger256([
                    0x392ebf3759ab18f1,
                    0xb5bdd0ed139059e8,
                    0x294c332ea2db11b7,
                    0x471b1ddddc1817d0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5147a56b7ad6e2f7,
                    0xb13832badac9a547,
                    0xbbaf72db8209a051,
                    0x259f272d63177dd3,
                ])),
                Felt::new(BigInteger256([
                    0x5ad4909fa12c4ea7,
                    0x552840b9ef92502b,
                    0x4301678ee4190a56,
                    0x1331502dbb612580,
                ])),
                Felt::new(BigInteger256([
                    0x6a55c0f7f3256478,
                    0x51e06bda85f8e4f2,
                    0xeee0b1a8843049e0,
                    0x0ea4fa7594ec9276,
                ])),
                Felt::new(BigInteger256([
                    0x597e6e4d821855d5,
                    0x8945c60fcdbb9062,
                    0xf1fed17e10d0113b,
                    0x64325f1fceff7c1e,
                ])),
                Felt::new(BigInteger256([
                    0x219b3bf2ec75e430,
                    0xf226fe499cbb639c,
                    0xd9ff00cdc3ec693c,
                    0x0f24ec1430521440,
                ])),
                Felt::new(BigInteger256([
                    0xa4c9fbd5f880578a,
                    0x9cc229a7b0e32af8,
                    0x8688eddaea2a1c6c,
                    0x49ce3bd5d92f05c8,
                ])),
                Felt::new(BigInteger256([
                    0x202f8e3696209cbe,
                    0x3a9b44f0a0f61707,
                    0x8267c94048c5a6dc,
                    0x30badbe37bb1bb0e,
                ])),
                Felt::new(BigInteger256([
                    0xad665fd7fc409f58,
                    0x75c89b682829fdcc,
                    0xee6f77b9b1f85ace,
                    0x0b6d639bf05eed87,
                ])),
                Felt::new(BigInteger256([
                    0x519d939829450dea,
                    0x1a3508aaeb729a2d,
                    0x29dddb742ac511e7,
                    0x4b0413f481ef937c,
                ])),
                Felt::new(BigInteger256([
                    0x9b3ce671ad281eb2,
                    0x78b54ecb166fe68d,
                    0x6aa9a7a23dabe81a,
                    0x3c5b7f194f835ff1,
                ])),
                Felt::new(BigInteger256([
                    0xcf55e092df52134c,
                    0x447c006a004d7110,
                    0xb9ed30aab609a99e,
                    0x4886d379aa76dc75,
                ])),
                Felt::new(BigInteger256([
                    0xb6bb0afc7de4deaa,
                    0xcfa1e02b03887bfe,
                    0x931223d9e74268a0,
                    0x6700ce908bc6916e,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x88cfc938d1d71e2e,
                    0x6a94959f88a03e84,
                    0xc024baa6f027eb76,
                    0x5a68aca042e0a410,
                ])),
                Felt::new(BigInteger256([
                    0xad7c7302a0c3c20a,
                    0x617614d682186224,
                    0xf9eec83461155ce7,
                    0x0d37496f64c786aa,
                ])),
                Felt::new(BigInteger256([
                    0xb99ad2ac54a1c766,
                    0xb665243707dfcf09,
                    0x09a0818a84d3e0f5,
                    0x58d4236e5adfff2b,
                ])),
                Felt::new(BigInteger256([
                    0x65de72349f2c9d98,
                    0xe3a913cc018bc692,
                    0xb4b4613688c01e63,
                    0x45abd2e01b6632aa,
                ])),
                Felt::new(BigInteger256([
                    0x354819ce8db92b28,
                    0x4c0f2ca6effe0b54,
                    0xd9ed484ee3cd8114,
                    0x0c5cf8a2d9556dda,
                ])),
                Felt::new(BigInteger256([
                    0x8ff3f61f5f1fe2b2,
                    0xc1fbe3f43cd81ae3,
                    0xf1abe3edcb8bc471,
                    0x00c76890816cb8fe,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0b4625ba573c54f6,
                    0x970300d331e86e0f,
                    0x731c8310b1ca0ec8,
                    0x54e95c33a941c8af,
                ])),
                Felt::new(BigInteger256([
                    0xbdcaea35a4366962,
                    0x45f348f1d1a211d0,
                    0xe306fb85518485fd,
                    0x46bb61319a134659,
                ])),
                Felt::new(BigInteger256([
                    0xf38e0159668305e6,
                    0x824d640997ad21ca,
                    0x151a2cfe26545149,
                    0x0e459b6bd15c1bdb,
                ])),
                Felt::new(BigInteger256([
                    0x81a2d003c7b0bb71,
                    0x69f23e559df61f12,
                    0x7d9925a0df45cfdc,
                    0x67095baf36b0c675,
                ])),
                Felt::new(BigInteger256([
                    0x2abf1e68166f8277,
                    0xf92ffc0bdf5056a4,
                    0x20273a07cbde31e1,
                    0x6029c9208392c5bf,
                ])),
                Felt::new(BigInteger256([
                    0x5971a5aa4932412a,
                    0x9926a7fe30dc910e,
                    0x63ea2ef66558c803,
                    0x2d63d79c0e52302e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9c30bb4855d9d565,
                    0x845a252676f1d73d,
                    0x81fb664008211c6a,
                    0x5bdba71bb08a83d3,
                ])),
                Felt::new(BigInteger256([
                    0x7be30e9f7e56d5f6,
                    0x1ffd59127816ae8c,
                    0x1cc88f20824aeced,
                    0x617c5eefa66fd04f,
                ])),
                Felt::new(BigInteger256([
                    0x3d1b9aff9c83a6ea,
                    0xc6566422a8154b53,
                    0x79debdeee27294ee,
                    0x72562b56d2376c4e,
                ])),
                Felt::new(BigInteger256([
                    0xf4e1721c384254d2,
                    0x44c0c5d644daa5c0,
                    0xd1e4dd35484ec8a9,
                    0x0b215ad43e260236,
                ])),
                Felt::new(BigInteger256([
                    0xc365612c850a145a,
                    0xe61c6fdefe8f082f,
                    0x5383154623d9e8c5,
                    0x530106b02650468c,
                ])),
                Felt::new(BigInteger256([
                    0x79398237201082c1,
                    0x24e7ea1f076b1c71,
                    0x8a5bd48514439be5,
                    0x10b2c2ab34bb30ad,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdaa655eaea2d2c6b,
                    0x9f5bdf317603d4d9,
                    0x7af4e0fecd130a6c,
                    0x418c7f0357480707,
                ])),
                Felt::new(BigInteger256([
                    0xf90080c29162593c,
                    0x377ba2bc2ccc0b27,
                    0xe5909e37ee612ce0,
                    0x532895c207cd3cb8,
                ])),
                Felt::new(BigInteger256([
                    0xddd64b8a564580d6,
                    0x6fa50f15e287ac82,
                    0x48f8f6ed1fee8054,
                    0x1eaabdf382c27a39,
                ])),
                Felt::new(BigInteger256([
                    0xad7a29d69f25bba9,
                    0xa0e4c9c7fe3fcbae,
                    0xc7a816dff20cec5b,
                    0x39b1c3ae909794fc,
                ])),
                Felt::new(BigInteger256([
                    0x144bf6a7da263375,
                    0x0b867033ae548711,
                    0xf1f042817b7869e1,
                    0x5e3bc71dbf3cc804,
                ])),
                Felt::new(BigInteger256([
                    0x97cc296014384a05,
                    0xe93eed8fdb65ab2f,
                    0x74f59097fbbc7f52,
                    0x51eaba756b8f60eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0efcbd06549d0a2c,
                    0x8717a61cb62dae81,
                    0xb5895a90133fc360,
                    0x0b6016ee604543e2,
                ])),
                Felt::new(BigInteger256([
                    0x717ede70bd85f6c5,
                    0x1c7d07cde801901c,
                    0x2a54feea675081ec,
                    0x510631a6747b4bbc,
                ])),
                Felt::new(BigInteger256([
                    0x9ce00364cdc2edd6,
                    0x6b0925e17ebb56a6,
                    0x6b2e20d27741552b,
                    0x0e0aa6e9bc7d860d,
                ])),
                Felt::new(BigInteger256([
                    0xa5a2e39f81b12322,
                    0x2a534af51d31b4b3,
                    0x91c8b2d59a3add98,
                    0x244c67d6bdf5a6cc,
                ])),
                Felt::new(BigInteger256([
                    0x5da1394293c957d1,
                    0xac6c143d68f2ce93,
                    0x389df954ab050aa5,
                    0x585e0997b95bfe97,
                ])),
                Felt::new(BigInteger256([
                    0x7f45654d31f6a359,
                    0x7499aef027b1ddaa,
                    0x3e294d4abcf94200,
                    0x524f1e5cb683a968,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd61f407f49546df3,
                    0x71c4c34b0f73051f,
                    0xc9388021f314bffd,
                    0x57d11a4df53da07b,
                ])),
                Felt::new(BigInteger256([
                    0x05c801cba0b356d6,
                    0x175398a409a7ea4e,
                    0x0f957ffd610f6365,
                    0x488b05d77b88f555,
                ])),
                Felt::new(BigInteger256([
                    0xf8411f77c573033a,
                    0x83b6fa19eadb5bb1,
                    0x504c123f704913fa,
                    0x1de93d3012399ed4,
                ])),
                Felt::new(BigInteger256([
                    0x4574ce21218bd98f,
                    0xd02123caccc38fc5,
                    0x2f38d1142afa4112,
                    0x21b12ab647ee7af8,
                ])),
                Felt::new(BigInteger256([
                    0x5ec67ed0f5e61fd6,
                    0x28e6583e08c0ab23,
                    0xac14ffb12db9fd9e,
                    0x4c7e6dcdf8c2ea51,
                ])),
                Felt::new(BigInteger256([
                    0x771819ced05a5dd2,
                    0x112468ea0f854210,
                    0xbfbaeec8d2b35c6f,
                    0x4635d5932d430d3d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x564bf45ec4f369bd,
                    0xae3d82163e30f77e,
                    0xaa975b5d559b2040,
                    0x351a2df804189287,
                ])),
                Felt::new(BigInteger256([
                    0xba11d289cf447381,
                    0xd50b73f26ba02205,
                    0xf57a25f69896809d,
                    0x3e8393f8146765f1,
                ])),
                Felt::new(BigInteger256([
                    0x77a9db04cf9c7c71,
                    0x0284a96982ae0238,
                    0x07db0018a8cf4c23,
                    0x11f9822ef911128e,
                ])),
                Felt::new(BigInteger256([
                    0x030319cb21ad389e,
                    0x3d5a93da716ebcc4,
                    0xbbbb97d789d8f424,
                    0x534d52d2a60c3777,
                ])),
                Felt::new(BigInteger256([
                    0x193a546c7ab4aead,
                    0xccf075532718acd6,
                    0xea68d3bd0c76d958,
                    0x5d5d0f6cfee52e94,
                ])),
                Felt::new(BigInteger256([
                    0x56fb6ab6da29401d,
                    0x85c6a473a4021506,
                    0xece2ded47c5b19fc,
                    0x5da294b0bcdd5f96,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb2a7fd99f7aee071,
                    0xb1afbe696a636a49,
                    0x41a4b0da8fa5b76a,
                    0x70c35ec34a36aa86,
                ])),
                Felt::new(BigInteger256([
                    0xba66600819fd203d,
                    0x12c3017c3c914e68,
                    0xd926d360c0c995bc,
                    0x094037af56601243,
                ])),
                Felt::new(BigInteger256([
                    0x247ca530ef38ca80,
                    0x109d608b4908409e,
                    0x75446ef8bfcf77be,
                    0x1e011658a5275b2a,
                ])),
                Felt::new(BigInteger256([
                    0x63fb6ded102a81a7,
                    0xe440686617af7b57,
                    0x794627b52c5df39c,
                    0x2bb85b72a4a64ccb,
                ])),
                Felt::new(BigInteger256([
                    0xda24b72df885bae1,
                    0x3538f3c2a43b618f,
                    0x0ed813e01b2731e4,
                    0x1019c9b23e83c0cc,
                ])),
                Felt::new(BigInteger256([
                    0xb6e0e7efbf29c0a7,
                    0x6972cf643c47bb2c,
                    0x3680a115a9e5e28a,
                    0x0ecc40976d3c92ea,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x46d1f90ea9f6a2dd,
                    0x3a89bad2de1380f8,
                    0x0520b011996a97f5,
                    0x10ff759ef3146849,
                ])),
                Felt::new(BigInteger256([
                    0x80f6cd4b811edb26,
                    0x59ce207000b8e6a0,
                    0x876fc43716c0af2b,
                    0x4113a77a58839d8d,
                ])),
                Felt::new(BigInteger256([
                    0x01be182d544c71e1,
                    0xb73c046957a1b313,
                    0xd61ac336aacd812f,
                    0x6e8df7f6c8dba873,
                ])),
                Felt::new(BigInteger256([
                    0xa3d9b9e086aadf06,
                    0x9080d09a59976d8e,
                    0x52102f0552b92f98,
                    0x526b39e3fb3fedbc,
                ])),
                Felt::new(BigInteger256([
                    0xff9b3e7a11df3110,
                    0x5b01394d08725052,
                    0x2ea6f1ea2ba1c9b4,
                    0x1ee4c7e8aa9d27b3,
                ])),
                Felt::new(BigInteger256([
                    0x06aa0976d4988b7b,
                    0x14ee8d1f73318473,
                    0x41d4f7264f1e8c6a,
                    0x36f246cb84015a4e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x606eb4675716e2f1,
                    0xc1ad074a7a1c98d3,
                    0x2104eac5480e782c,
                    0x4e256a5236bcbc97,
                ])),
                Felt::new(BigInteger256([
                    0xc5fb495e677dd81b,
                    0xe07c74ea0c237a20,
                    0xd2cbc5a4aa178c2f,
                    0x4fbc581a85ec8da6,
                ])),
                Felt::new(BigInteger256([
                    0x315355b1f0a82c8c,
                    0x21a75fb290bb79df,
                    0x892a582b906838b3,
                    0x48ae0280548546b9,
                ])),
                Felt::new(BigInteger256([
                    0xe6e03b53a2902dc3,
                    0x8fb02ca414eabbfe,
                    0xf383ddbf201a237e,
                    0x42e04518cc0ff794,
                ])),
                Felt::new(BigInteger256([
                    0x0321b3be4d971af0,
                    0xf52cbe7b6f69db37,
                    0xf2dd111e2d70bdf5,
                    0x3f47bc16306e3330,
                ])),
                Felt::new(BigInteger256([
                    0x87dbf1aa4c20ea04,
                    0xaec1d243a9a37112,
                    0x0b93f388336716a7,
                    0x48b59d8ca36009db,
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
            vec![Felt::zero(); 12],
            vec![Felt::one(); 12],
            vec![
                Felt::zero(),
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
                Felt::one(),
            ],
            vec![
                Felt::one(),
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
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x66beb0b22c44b6c6,
                    0xbc283ad3298058f6,
                    0x7dac9ceecbc701ac,
                    0x3a38c2a22fbc9783,
                ])),
                Felt::new(BigInteger256([
                    0x31875647a029fb8a,
                    0x61bfd1504f1b5b39,
                    0xaf0280adea30a123,
                    0x48d8fe6061b7a0c5,
                ])),
                Felt::new(BigInteger256([
                    0xde504b4c1c856194,
                    0xc59b442ba85e6c4e,
                    0xff28ff79fdbb3a75,
                    0x26bebc4247e5ad6b,
                ])),
                Felt::new(BigInteger256([
                    0xe8cd29a7d33e3af9,
                    0xb169e99b33fe599c,
                    0x7793bd4d72afe6c9,
                    0x18ef0a4d2791d349,
                ])),
                Felt::new(BigInteger256([
                    0x9082debb72eaaaa2,
                    0x85f5b6b08fc9881f,
                    0x3956f7bc17122075,
                    0x3e80daca7cd160de,
                ])),
                Felt::new(BigInteger256([
                    0xe89119f2f14d9ff0,
                    0xfcc9715424640957,
                    0xb24dccf5d258f680,
                    0x1ca8e070a5792118,
                ])),
                Felt::new(BigInteger256([
                    0xc814abbf2b0cbb98,
                    0xfa38c1b3469cff5e,
                    0xf0cc43e0718bcc20,
                    0x53f1284af4577c5b,
                ])),
                Felt::new(BigInteger256([
                    0x052e597ad73e48b6,
                    0x4016e347aa218020,
                    0xd278359f1bc9df37,
                    0x4b007a0154b3ed04,
                ])),
                Felt::new(BigInteger256([
                    0xaf7015212c56bac0,
                    0x953e3eb059025e71,
                    0x202759a4dc997348,
                    0x2d72e8056b783e9a,
                ])),
                Felt::new(BigInteger256([
                    0xd66f66ec6af448d7,
                    0xed33f9e5f559b566,
                    0x21d3f97d32917fdb,
                    0x60d84c05aa572b17,
                ])),
                Felt::new(BigInteger256([
                    0x893f81be7a19b0c7,
                    0xa430734cce5de5b4,
                    0x4b88b2bb34f8a6c4,
                    0x35ce66eb5659327d,
                ])),
                Felt::new(BigInteger256([
                    0xf500ec73ee505e3d,
                    0x7a7dbec1402302e6,
                    0xf05124855c323154,
                    0x211a3db2da24ee3d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x55493ecf2e846eae,
                    0x3cd63a6f43479bee,
                    0xa560db7902740e9f,
                    0x40c90f221951626b,
                ])),
                Felt::new(BigInteger256([
                    0x9e1e29c8cd0b2322,
                    0xf2389a8119c59010,
                    0xc249a9d13849ba23,
                    0x3874bd006a798ff8,
                ])),
                Felt::new(BigInteger256([
                    0x2bbebfe0eca4f2e2,
                    0x3922294c6d3c6e80,
                    0xcbc2a1e6e11c314c,
                    0x12c01b13dfa211e4,
                ])),
                Felt::new(BigInteger256([
                    0x2a21be821a0814b6,
                    0xc34d5c3a2862c065,
                    0xb42fe88899b9733a,
                    0x4762fc8c4277a8fd,
                ])),
                Felt::new(BigInteger256([
                    0xc1d57d1387c299d5,
                    0xb2965a439eabe681,
                    0x4125297e30d47bc1,
                    0x5808e2e85860f9c0,
                ])),
                Felt::new(BigInteger256([
                    0x5cec45f3417da80c,
                    0xc93bddd0b4936016,
                    0x716698bee7489244,
                    0x5ad0b6642a077c6c,
                ])),
                Felt::new(BigInteger256([
                    0x2ab57a86f9e324bd,
                    0x1a13ab5ebee0cb70,
                    0x820bff6aa65cdc14,
                    0x07a8e77d9af80169,
                ])),
                Felt::new(BigInteger256([
                    0xb0ff641e1b784fde,
                    0xc6438031d6175a9a,
                    0x1069548b6aa7af8a,
                    0x5a4cb27f62f214b0,
                ])),
                Felt::new(BigInteger256([
                    0xe550d3fcf875bd97,
                    0x792056bfb874c2f0,
                    0x66deb32b169ce2a5,
                    0x35ba7a422f08bad3,
                ])),
                Felt::new(BigInteger256([
                    0xbff73a57846c81fa,
                    0x9abaf5c2926e4ea3,
                    0x60e9a6cac361e638,
                    0x61b15da2f1f9b1c6,
                ])),
                Felt::new(BigInteger256([
                    0xc5a13c2cf26167c6,
                    0xa39c6b536791d462,
                    0x89c15a31c9c4ffb6,
                    0x3ff1f834415ad4f5,
                ])),
                Felt::new(BigInteger256([
                    0x7a7c305fc3aa25ae,
                    0x8fdb1b7ffa4b9b1f,
                    0x1efb245aa164d994,
                    0x6f6e0245434c9714,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x523d2cd7e8467c57,
                    0x9f714fb0865a4a09,
                    0x33c7477446812c2b,
                    0x59de5b1e6aa9fa1e,
                ])),
                Felt::new(BigInteger256([
                    0x57fe657db59ee20a,
                    0xba5b16b918b8cea8,
                    0xcdbdc59cb90e61d9,
                    0x17b24d7fa5112ec9,
                ])),
                Felt::new(BigInteger256([
                    0xfd33fc88c1d31355,
                    0xcff46f2ddc42b1ae,
                    0x37aac48b5a4400ac,
                    0x2a66486a46d08016,
                ])),
                Felt::new(BigInteger256([
                    0x342a694982011fbe,
                    0x8bf371c935e91b70,
                    0x2220b80470c2e3bc,
                    0x51aca70d011e02f4,
                ])),
                Felt::new(BigInteger256([
                    0x774aad362cd6d907,
                    0xc374e31fea1cdfed,
                    0xa7eb52742eda81b1,
                    0x714dfd6868da0947,
                ])),
                Felt::new(BigInteger256([
                    0xa8f8f4e555bce798,
                    0xea02fd0717d00d6b,
                    0x310f1e9c9e588b58,
                    0x386805e874ff0024,
                ])),
                Felt::new(BigInteger256([
                    0x0fd0e98717ee4efe,
                    0x52cc64e155758bb1,
                    0x66c131dccff1d6ef,
                    0x4ae97dd5dc8fb4aa,
                ])),
                Felt::new(BigInteger256([
                    0x0aca8a3a018e2b5e,
                    0x38b89221b135d3ba,
                    0xcc25b49c7c6a4faf,
                    0x583973c27d0a8d4b,
                ])),
                Felt::new(BigInteger256([
                    0x412fb13ab4fe4682,
                    0x7b34ee0477ecfb2a,
                    0x29fe41dc3877e906,
                    0x27426450114b4085,
                ])),
                Felt::new(BigInteger256([
                    0x70af33f61037eabf,
                    0x1ebe4bb59a6b8ee0,
                    0xa8f6ae462dfeb822,
                    0x32ddadf4f36d86a8,
                ])),
                Felt::new(BigInteger256([
                    0xe6ae2898605652bd,
                    0x98874196bc1f301f,
                    0x6f394969640bc087,
                    0x035d5b0c102735a2,
                ])),
                Felt::new(BigInteger256([
                    0x08658d1a54cc9752,
                    0xc219e3d01a1b64d5,
                    0xc03cc7c22ac3e0a9,
                    0x5b2ec278fa547a36,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x71f60e2790380604,
                    0x86598eada240c289,
                    0x97f1033d6b821c67,
                    0x31effcab9c9d20a6,
                ])),
                Felt::new(BigInteger256([
                    0xd6974a7799bd1dab,
                    0x195f5cea6c3327b3,
                    0x32b6e6ec3feed376,
                    0x480aff8f50a546ce,
                ])),
                Felt::new(BigInteger256([
                    0x50bea12335dec642,
                    0x7f3d397ef4082f6b,
                    0x2e92b08c3ff74bec,
                    0x0e06ceb2a728b414,
                ])),
                Felt::new(BigInteger256([
                    0x107fb613c754632d,
                    0xce96048fa03bf358,
                    0x7e45bb5b7e977790,
                    0x58f42bea91a18347,
                ])),
                Felt::new(BigInteger256([
                    0xb82feec9007cd0b6,
                    0x76f638ecc4df17d1,
                    0x82cda48a906892fb,
                    0x4f61e1173699db04,
                ])),
                Felt::new(BigInteger256([
                    0xcb256804451865cf,
                    0x28a35a07308ff742,
                    0x0c8beae7df763e43,
                    0x3a3db1ddf49d44e8,
                ])),
                Felt::new(BigInteger256([
                    0x6dec93e933afe5d5,
                    0x0cd645d1e337274b,
                    0x693013cb2c95c94a,
                    0x00686b004c1a77e1,
                ])),
                Felt::new(BigInteger256([
                    0xa2cc35c401c37f82,
                    0x9f06518a7c634759,
                    0x5ad5c31feef7e228,
                    0x3d339a1186939ef4,
                ])),
                Felt::new(BigInteger256([
                    0x31cd931e713fc45c,
                    0xb0ea1678d8590e5e,
                    0x3dc121182c85a141,
                    0x3f73d9e149b0cd37,
                ])),
                Felt::new(BigInteger256([
                    0xea5712f784debf61,
                    0x510d5e765c886536,
                    0x788df1bf4210c425,
                    0x41005171c9dc08b5,
                ])),
                Felt::new(BigInteger256([
                    0x6c6e0b6daa58fa32,
                    0xd6a0740f4bada4c4,
                    0xdfcd70318c5ee17e,
                    0x327b75a0eba5cd43,
                ])),
                Felt::new(BigInteger256([
                    0x2050e771a38791ac,
                    0xe6537b8f3aa09b0d,
                    0xd0f23d0792e62be8,
                    0x43885f086211eaf3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x16bfb86cb238fcb9,
                    0xd37236082cd364a9,
                    0xc726cd7ed3fddd13,
                    0x37d66b706255d6be,
                ])),
                Felt::new(BigInteger256([
                    0x62830e1bdeda4067,
                    0x74f5b1d6f4a427d2,
                    0x9fac68d7634bd6ab,
                    0x5cd85ae96e1d08f5,
                ])),
                Felt::new(BigInteger256([
                    0x529535d5b09e1a3a,
                    0x0b6697ecff825207,
                    0xc8b36ea2d64343a1,
                    0x1e8ef5f6e961af82,
                ])),
                Felt::new(BigInteger256([
                    0xdd37b500977dee91,
                    0x7ce35296ef3bec3f,
                    0x9b821c348b33b9fe,
                    0x48cde0efa99c5174,
                ])),
                Felt::new(BigInteger256([
                    0x26ddfebe946b4d5c,
                    0x19c91c5d4e0e5792,
                    0xafe14e2175aba3e2,
                    0x264e71b6e981543c,
                ])),
                Felt::new(BigInteger256([
                    0x71e32cec533eb5e2,
                    0xab783998a31308ff,
                    0xa5544f211c619a4c,
                    0x5bda3880f8519f70,
                ])),
                Felt::new(BigInteger256([
                    0xeb5655c71a4215d6,
                    0x8a209c9445dd8173,
                    0xd7f461cdfad0801c,
                    0x3dc5c9dbcdfbdd9d,
                ])),
                Felt::new(BigInteger256([
                    0x5f986fff04c4efcf,
                    0x45411a830a81f508,
                    0x315a22b644d7ba01,
                    0x415da349f7b1cd3c,
                ])),
                Felt::new(BigInteger256([
                    0xbe55af7995a8182b,
                    0xaac1bdd409cf3a9e,
                    0x35431fb68af67f6a,
                    0x66025b62883a9665,
                ])),
                Felt::new(BigInteger256([
                    0xfcce86e70c893fbc,
                    0x304b61d48fe1c009,
                    0x4c77587f613b2e63,
                    0x11512cf19194bd18,
                ])),
                Felt::new(BigInteger256([
                    0x0a4e860d1be66bfe,
                    0x62faf3bd7ffbcc29,
                    0x82cdf94a9260433e,
                    0x51b5aebd9b3e5b37,
                ])),
                Felt::new(BigInteger256([
                    0x5ebbea55e273f434,
                    0x1891f0fb9c3fd4dd,
                    0xd0187fc0b9306e55,
                    0x654cd2ed933e2f9d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x708b70beba1a12d3,
                    0xce3393a8050dce53,
                    0xf56de012e3ca1e40,
                    0x165f1047d00d67a1,
                ])),
                Felt::new(BigInteger256([
                    0x9b17be5c299f5cf7,
                    0x8fdd42f388d96c72,
                    0xaf5b14b161582f77,
                    0x289d67077816b232,
                ])),
                Felt::new(BigInteger256([
                    0xce31ebf93e0af8f3,
                    0x609a6c454aac1338,
                    0x4167c6183dbdeb30,
                    0x1bb8372bc60df635,
                ])),
                Felt::new(BigInteger256([
                    0x6684c28667b0a8e9,
                    0x0aa1cf47e288f05b,
                    0xd9d186f5991b5555,
                    0x0d2bcf805ffb0e35,
                ])),
                Felt::new(BigInteger256([
                    0x2fe9df7dd023f96d,
                    0x02e23638e30b5251,
                    0x44ce0441958ce91a,
                    0x34fc357902ddec30,
                ])),
                Felt::new(BigInteger256([
                    0xe183df9218c80cfe,
                    0x3f7126bd89f5e2ba,
                    0x6da6e517356a21d1,
                    0x544331b0db6c90e2,
                ])),
                Felt::new(BigInteger256([
                    0x72b635f4e869c9d7,
                    0x4b055885d2a31ec9,
                    0x5917fc257755a920,
                    0x49ed5f7fb0bf20bb,
                ])),
                Felt::new(BigInteger256([
                    0xae257b0411fa9cd9,
                    0x5df8983197151838,
                    0x7eef3dbfa1b3693d,
                    0x06b02f26e18b01f8,
                ])),
                Felt::new(BigInteger256([
                    0x713468df7231c092,
                    0x26f8e659393e384c,
                    0x8b42cfbc1d70cbca,
                    0x0bca858f4a15e8da,
                ])),
                Felt::new(BigInteger256([
                    0x73f14d722bb84210,
                    0x00f9506172416dcb,
                    0x453cf1ea0c1eb050,
                    0x4598602d07e35dc4,
                ])),
                Felt::new(BigInteger256([
                    0x9fa1cc5bbb5a7188,
                    0x754e5e2591330fd2,
                    0x58a2cc74393e3465,
                    0x6bd85911c145bc08,
                ])),
                Felt::new(BigInteger256([
                    0xafd003ae1abd8c8d,
                    0x61c36909645adf3c,
                    0x995c1f237c6d5568,
                    0x07ea71dc66165939,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x77b2b5b4b43210bb,
                    0x194b427a807fbce3,
                    0x7078ac784f27757b,
                    0x4bac215e4d7893ce,
                ])),
                Felt::new(BigInteger256([
                    0xa34edb569f104254,
                    0x071b0c96c07c439a,
                    0xa04f0d58b5613fbd,
                    0x53aa84e0019a7254,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2993457cd42edd52,
                    0xbec2bce5a8e78a7f,
                    0x7524120e9a5ab9ee,
                    0x4f6b196cd4932d01,
                ])),
                Felt::new(BigInteger256([
                    0x98df5fe4b51965fc,
                    0xf54e8b42a07665f2,
                    0x915078148c8145d7,
                    0x673aed29b578bfb5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9cb1b776776790a7,
                    0x8951b1221d9972c2,
                    0xe8e98964fb29ea14,
                    0x39578a7c55d73c1d,
                ])),
                Felt::new(BigInteger256([
                    0xe9fe02f3d6a9ad88,
                    0x35e86504c45e14bf,
                    0x45cf68d2d53b7976,
                    0x0962d51befb385eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xccc8981e1a98e0b5,
                    0xc6c9ba7806e1ac6e,
                    0x82a442655ed81c9c,
                    0x4a855cc16fa9cbfd,
                ])),
                Felt::new(BigInteger256([
                    0x3e46d3fa44c05ee9,
                    0x6de1b61106732607,
                    0xeef46da7d288c089,
                    0x6ad76c92da56b558,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf6f7aa12542fa27c,
                    0xa9c684f6214e2900,
                    0x7f279f79dc0a33d6,
                    0x0c65beb553f93894,
                ])),
                Felt::new(BigInteger256([
                    0xe9563c1355a2eefd,
                    0x3e4ecf25016d3909,
                    0xc55f3fe2416b30f1,
                    0x57ed0a45f8201b64,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc6b0e915314b0755,
                    0xa47202e4edea42c2,
                    0xf81cb9e170dd8091,
                    0x4fc945e970570c2d,
                ])),
                Felt::new(BigInteger256([
                    0x48dcd618a30e5e88,
                    0xb7e5cd9c62ae33fe,
                    0x6350280a8c267edf,
                    0x1233292562933495,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8aeb810cfff76608,
                    0x65439b9ba24e0d67,
                    0xaf33b58fc2a77902,
                    0x23cd685bcfceda02,
                ])),
                Felt::new(BigInteger256([
                    0x484d925e426f20f6,
                    0x9e89b595ca4e447a,
                    0x04a6fcae258ad2c3,
                    0x614371d182e6b4e1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x41b6300a8b460e6d,
                    0xdc18635b055368c7,
                    0x1597c1b51825c9bf,
                    0x4713394e2b9fe07d,
                ])),
                Felt::new(BigInteger256([
                    0x44c1fd07fdc1153c,
                    0x3767b934c4b95a0f,
                    0x264f98d339c81a29,
                    0x33e5c57538e2f538,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x46778d15f783f6cf,
                    0xa242b4520f564fac,
                    0x0412d44eea7d9ddc,
                    0x690f845a63a89875,
                ])),
                Felt::new(BigInteger256([
                    0x1c8091aa4a6fc8ed,
                    0x07df4b30a1cc83ce,
                    0xd2031f463056ab15,
                    0x1d518e598c2f6c7c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2ddf651cefdb5d3f,
                    0x590733aba1e73e80,
                    0x1d6a0f201a08f048,
                    0x0c127c744806b222,
                ])),
                Felt::new(BigInteger256([
                    0x7e1f903457fc8ca9,
                    0xadfce4dd07739954,
                    0x40d8453eff7fcade,
                    0x6554664990ab47e9,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
