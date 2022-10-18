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
                0xb2ecfe19ffe61bcb,
                0x601e03aec855ed25,
                0x66ff571c17a10320,
                0x317cd171990f269a,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xb7f9105124dd0bf5,
                    0xec243a4dc307d6a3,
                    0x5c2550f76e27ae29,
                    0x383b5cd7da891dc5,
                ])),
                Felt::new(BigInteger256([
                    0xa774fb87f52366a6,
                    0xcc9760ea9181b05a,
                    0x11050573b7b7f4cf,
                    0x208b4cef68b14c86,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf7e4ce0bfda3116e,
                    0xb352c4ec3eddc9dc,
                    0xc415fbec79baee27,
                    0x040179c013c44c8b,
                ])),
                Felt::new(BigInteger256([
                    0x04de9f7c8719426c,
                    0x473abef30d903f64,
                    0x588305db5037cb0a,
                    0x0fe625c536b6a2e3,
                ])),
                Felt::new(BigInteger256([
                    0x573db14e60ba4d76,
                    0xef7f9e63a7a1d1fb,
                    0xfdc3a0456e88978b,
                    0x1088229526bc5eaa,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3139ad2dd1c13b7e,
                    0x03c7f5f2456175d7,
                    0xb19c16946a9193bd,
                    0x2b81313eb90cded8,
                ])),
                Felt::new(BigInteger256([
                    0x83e62c6db2c1bbb7,
                    0xd2a5b785dbf0017f,
                    0x88d167fec1e5b12e,
                    0x1575252efe59d13b,
                ])),
                Felt::new(BigInteger256([
                    0xfa38092f1a62f93b,
                    0xa334c07c4b6f05a0,
                    0x97ba497e79f77fc3,
                    0x36a6acc6d742d86c,
                ])),
                Felt::new(BigInteger256([
                    0x5bdef9ca63ebe737,
                    0xb021dcca0d818272,
                    0x4a9cfcda86acd87e,
                    0x1b9ab7a80329cad6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x92b740e2bb65fd66,
                    0xf3ed1de2e43938d8,
                    0x89345405bfa90bf0,
                    0x21c5bb86927b0d91,
                ])),
                Felt::new(BigInteger256([
                    0xc80a9437eb21701b,
                    0xd6af360c2261c65a,
                    0xd179b793261613ce,
                    0x387d286beb89d458,
                ])),
                Felt::new(BigInteger256([
                    0x0cd29b400efd52b6,
                    0xacecac14cc9b1c4c,
                    0x06d71c890c22b328,
                    0x14ccd78d872b7886,
                ])),
                Felt::new(BigInteger256([
                    0x57935c4f9ea765ae,
                    0x442b34778b63f65d,
                    0x2b85cdd177365915,
                    0x1523eb490a5c5cd5,
                ])),
                Felt::new(BigInteger256([
                    0xa3b5d0c7fb5ada3e,
                    0x6d2ba6028523d7a4,
                    0x71d3aa7d0a985177,
                    0x37d544ee48e41ccc,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4afa9170f7aabed4,
                    0xe8233b88e8a5a5bb,
                    0x4e9451b323fb6dcd,
                    0x047d3cf080b383f6,
                ])),
                Felt::new(BigInteger256([
                    0x29ed7ad6b8b63544,
                    0xc2748f614e93f0b7,
                    0x8e90502f5e940d2f,
                    0x35e1982221264356,
                ])),
                Felt::new(BigInteger256([
                    0x8e3c59c6261c0db3,
                    0xf4dd367dbb9c8616,
                    0x66a0f68287475ce8,
                    0x0e4458ddc4897952,
                ])),
                Felt::new(BigInteger256([
                    0x6f4d10e0811356b3,
                    0x7238d7a832cbc88d,
                    0x248176ef2750f1a0,
                    0x11763af0acff19c7,
                ])),
                Felt::new(BigInteger256([
                    0x10f7ded110ced747,
                    0x3a8be1c4bc1ae177,
                    0x9be7a48c4dabc419,
                    0x24f12f4028dcf284,
                ])),
                Felt::new(BigInteger256([
                    0x8cc5c4eec896d371,
                    0x1b500f4f6aef38cf,
                    0x13cafd9bebd90b9d,
                    0x18d892488570b35e,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x0cb1efa6f7615257,
                0xa35d6e51f7b616f5,
                0x1a3a5c43d03d0a61,
                0x05ccb568ac20e6ba,
            ]))],
            [Felt::new(BigInteger256([
                0x843a4a4642d1d026,
                0xa2d89f6e48b7d17f,
                0xcb0a7e72892f7f9b,
                0x291eb2534daf0af1,
            ]))],
            [Felt::new(BigInteger256([
                0xd5cd3da4fdcaa5a9,
                0x6112b0542ed28ea2,
                0xa81ce7d22fa46196,
                0x3331c0282586f6f5,
            ]))],
            [Felt::new(BigInteger256([
                0xbb8a8a49e22418c4,
                0x7293888878217348,
                0x2a5f755652f631f9,
                0x22930b543688f6d4,
            ]))],
            [Felt::new(BigInteger256([
                0xc0c5f3a3b7bc7d46,
                0xa7c05d08601d8f94,
                0x70ea56a1bf226271,
                0x0f80ca6b8fb3556a,
            ]))],
            [Felt::new(BigInteger256([
                0xd456f908ff24722b,
                0xf60edc6bed84c346,
                0x39e505af21cd8b7b,
                0x2b44d1d0263960bf,
            ]))],
            [Felt::new(BigInteger256([
                0x5f097f55cc1bc0e9,
                0xc5994f9f45af384d,
                0x76c74a8fe678fe31,
                0x3f6acc0d0ed2d8b3,
            ]))],
            [Felt::new(BigInteger256([
                0x2bb3ccaba95cc58b,
                0xeb90b78c05032301,
                0x1514405f30318e83,
                0x05544e85f718a352,
            ]))],
            [Felt::new(BigInteger256([
                0xbdc7125850a88f5b,
                0x19309414e799e3dd,
                0xd822271d224775be,
                0x2a9296683a6dd02d,
            ]))],
            [Felt::new(BigInteger256([
                0xae51d58d0a364f44,
                0x3848cc97b53974c4,
                0x3584a76e90618851,
                0x1a27e83990b5a234,
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
                    0x5b34f0e417c45573,
                    0x59ee4e1000939071,
                    0x71c2514ba48b0e7b,
                    0x2a19830ac50209d5,
                ])),
                Felt::new(BigInteger256([
                    0x8bbf1e5d755a801d,
                    0x0def0e280ea5b33c,
                    0x71d766341a686153,
                    0x20a599ba88ec9161,
                ])),
                Felt::new(BigInteger256([
                    0xb6470f7250dc57e2,
                    0x68ae4071534a3b18,
                    0x6e768e0a81b99792,
                    0x380156aee0f2c494,
                ])),
                Felt::new(BigInteger256([
                    0xd7a986716e486949,
                    0x048f04070ca2b6cf,
                    0x2f36e24bae8c5c25,
                    0x331960f0dc85d5f9,
                ])),
                Felt::new(BigInteger256([
                    0x581c5157d2e94010,
                    0x8065a58d8b1531be,
                    0xa342697eafefa5e2,
                    0x1ed11988c9a36bf9,
                ])),
                Felt::new(BigInteger256([
                    0x16e8164671640c1a,
                    0x6670eb08a9a72194,
                    0xe2e0dc7aabbaa2a2,
                    0x1d655170d67dd893,
                ])),
                Felt::new(BigInteger256([
                    0xa1db3a31331ad1ee,
                    0x41605884fd6b5744,
                    0x159e2cbbe3b2f685,
                    0x3f661037234a9fc5,
                ])),
                Felt::new(BigInteger256([
                    0xb2babb696566b8c6,
                    0x618685e3e3f4c7c9,
                    0x1d74e50247e8dbf3,
                    0x2778c8b9689537db,
                ])),
                Felt::new(BigInteger256([
                    0x034722507e210779,
                    0x96b34c67a7b3cb9f,
                    0x9c50469c026dbf91,
                    0x04d55f7d86641de7,
                ])),
                Felt::new(BigInteger256([
                    0x42a4b154f0a01468,
                    0xbef6ddded20dbf84,
                    0x2b2c3ba9dc2a6de4,
                    0x0340e8ab6fa7d0b2,
                ])),
                Felt::new(BigInteger256([
                    0xc17e82e2fb849391,
                    0x62e9a8d5c1affd87,
                    0x4899408017feacf8,
                    0x2a1ad7bc1adccd04,
                ])),
                Felt::new(BigInteger256([
                    0x4acbc747e574c21b,
                    0x5a67f0d25444b14e,
                    0xca840012b995c590,
                    0x345d51bb0e6242eb,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xccf20414e2f2ee01,
                    0x7dbd91f673e8e6e9,
                    0x348ecc57e8f4c3d4,
                    0x3865f646a063efc2,
                ])),
                Felt::new(BigInteger256([
                    0x34caf8ff165d3102,
                    0xb2057b66b1a1c79b,
                    0x883c299d24f05c5c,
                    0x0328e62426761b1c,
                ])),
                Felt::new(BigInteger256([
                    0x64ebc483ff4a0c3e,
                    0xdd859729c07cf91c,
                    0xcf037c28c8977c01,
                    0x195c7d88443c37d1,
                ])),
                Felt::new(BigInteger256([
                    0x0d1231598595d578,
                    0xb1a47a21d3d13272,
                    0x73c3696603e5d918,
                    0x0f799e9061983860,
                ])),
                Felt::new(BigInteger256([
                    0xdd49660c9f7a4093,
                    0x1db04bea93d73f6b,
                    0x5c395ee6f4fcaac3,
                    0x00df98bb85b277d0,
                ])),
                Felt::new(BigInteger256([
                    0x1f6962caa49c6616,
                    0xf8ccb2d3ace0ebcf,
                    0xe56dcc655188e706,
                    0x139d5281b6d15521,
                ])),
                Felt::new(BigInteger256([
                    0x5bced3cfe90dac4f,
                    0x3a15176ddfb83bd8,
                    0x041b122990c9f00e,
                    0x1da1c5421b0a7f87,
                ])),
                Felt::new(BigInteger256([
                    0xff7e5ebee6feb94c,
                    0x552136cfd8963f85,
                    0xc613a24ad2ab52ad,
                    0x374b809b6b4944d8,
                ])),
                Felt::new(BigInteger256([
                    0x2781a0e960e958d7,
                    0xb802ab0a66c24a04,
                    0x22884d88e7837633,
                    0x36903524559b89ee,
                ])),
                Felt::new(BigInteger256([
                    0xd8a8810987ba96c0,
                    0x360c87d79b645f24,
                    0x77377907df0c7c14,
                    0x3551f62457f7f650,
                ])),
                Felt::new(BigInteger256([
                    0xe75db304995bd71f,
                    0x78aee17461f62729,
                    0x47b58af110942bc6,
                    0x34f2e370f46b82f0,
                ])),
                Felt::new(BigInteger256([
                    0x6e16f56344f292fa,
                    0xaba37d65eaa45092,
                    0x1c11af25f4ed5706,
                    0x3bce03763cd5f9f5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2dda778f6b82896b,
                    0xe04e7869bf16afc6,
                    0x517a6f3801b20168,
                    0x163387787e1edcbc,
                ])),
                Felt::new(BigInteger256([
                    0xbe5a753ca6c30235,
                    0x9c6cbaf67df6ac4e,
                    0x24713de57343fa36,
                    0x2e149eb684d29379,
                ])),
                Felt::new(BigInteger256([
                    0x212bdf91dc3340f6,
                    0xb11f1c24eed5c20d,
                    0x5a9efda9fdf2d3a1,
                    0x3b8ee236ae0f5337,
                ])),
                Felt::new(BigInteger256([
                    0xcbd45763f68671f1,
                    0x567777e64ce62848,
                    0x5fcd4d968653a78d,
                    0x051d541817ba1adb,
                ])),
                Felt::new(BigInteger256([
                    0xda140bb1f1efd0a4,
                    0xb8daaf1b5e43d693,
                    0x80fdee5b33780277,
                    0x38b13d992382700b,
                ])),
                Felt::new(BigInteger256([
                    0x9f5a71b3db40bc91,
                    0x4134cf99f3dc0dcf,
                    0x5cbd424622c88bd0,
                    0x2b6a0831cf426d00,
                ])),
                Felt::new(BigInteger256([
                    0x42a6adaef7ab1c24,
                    0xbabf0c665323716e,
                    0xb88b6ebda0f77d07,
                    0x1709b9f5ba9ffccb,
                ])),
                Felt::new(BigInteger256([
                    0x5ffefe8cf3c2751d,
                    0xcf70480c220163a9,
                    0xfce7fc51850e9e2d,
                    0x2af3d8152e2bbb9a,
                ])),
                Felt::new(BigInteger256([
                    0xe1d423f7fd821768,
                    0x69d0d59e0216d390,
                    0xdaf22d58d62880aa,
                    0x2082d8ae7c77a8dc,
                ])),
                Felt::new(BigInteger256([
                    0x4a9d05724ebd7f89,
                    0x07d3346f7fa478cf,
                    0x99716f73fe8636b1,
                    0x1e077e305338740e,
                ])),
                Felt::new(BigInteger256([
                    0x2d8af6b11551f3ad,
                    0x01e0991e9f821792,
                    0x5887ade6d1938cb0,
                    0x0782c1d91c014533,
                ])),
                Felt::new(BigInteger256([
                    0xb95e3be72ffa4438,
                    0xc2dcde55ca56a304,
                    0xde171c76a25f1b5f,
                    0x060240d8ddcf7e22,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0ab18d46f88d96a7,
                    0x1d574d59a7c41c02,
                    0x1253292f65f52404,
                    0x3c3834c0eab011f6,
                ])),
                Felt::new(BigInteger256([
                    0xd7f25282f79ce760,
                    0x4f436bede03619c7,
                    0x65b31380c36b06f0,
                    0x3841f88655530078,
                ])),
                Felt::new(BigInteger256([
                    0x686057469c096cf7,
                    0xfd49fb8cc8952d2c,
                    0xdfee6deaea353543,
                    0x2517afcb0ebf368d,
                ])),
                Felt::new(BigInteger256([
                    0x3a8c47603295d40d,
                    0x4d9f8c5fa6566b9d,
                    0x62ca6d2e44cfb3c5,
                    0x16c605576219b7c8,
                ])),
                Felt::new(BigInteger256([
                    0x291dfa73d861f12b,
                    0x8601bd5adb3babc0,
                    0x6bf97141ed3a7420,
                    0x0065288f658a1426,
                ])),
                Felt::new(BigInteger256([
                    0x2e092187ee7e163b,
                    0x89dba94c5e31de19,
                    0x5b4dab046fdaeb61,
                    0x027489326a3306e8,
                ])),
                Felt::new(BigInteger256([
                    0xe0b61cb320f32f39,
                    0x43f1d6515939b792,
                    0x17c4c02c43c7845c,
                    0x241de24eede4a810,
                ])),
                Felt::new(BigInteger256([
                    0x1a5bc58830ea2eb8,
                    0x22f095a9bbda0d29,
                    0x0122cddab508439d,
                    0x3292e5f4b746df06,
                ])),
                Felt::new(BigInteger256([
                    0x7ad0002d136ca9ff,
                    0xcea518a01061763a,
                    0xace51b08a3aff561,
                    0x0d9b64c565800a67,
                ])),
                Felt::new(BigInteger256([
                    0xe69fa96e3607e9e5,
                    0x96cdeebf9c4d33a8,
                    0x87b017742233b1f8,
                    0x16be96c10a8dcc4a,
                ])),
                Felt::new(BigInteger256([
                    0x4cb57856856389ab,
                    0xed907199b72bc57c,
                    0xd189638cf063086d,
                    0x13882a2f5e1ae0a5,
                ])),
                Felt::new(BigInteger256([
                    0x9e74c1dfe04255fa,
                    0x7bf48e27bb4070f1,
                    0xe0156addbb885434,
                    0x200a24597ac8903f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x14f0f2f580732cdc,
                    0x4a6c06fd132b8ef7,
                    0xa2f77ff7d98ba868,
                    0x0eaddfd6de270cc6,
                ])),
                Felt::new(BigInteger256([
                    0x5c14e1574dc0246d,
                    0x366fbe711f91ca26,
                    0x34a41e89c2550789,
                    0x0af0f358678c6a57,
                ])),
                Felt::new(BigInteger256([
                    0x84e03c5f4a24d7a9,
                    0xf0efaae546632fc9,
                    0x29deb8dfb4135cf2,
                    0x29697417cb603190,
                ])),
                Felt::new(BigInteger256([
                    0x8b9c5158e0b30fdd,
                    0xfaf888fee8b05cb5,
                    0x87b58826517fe0ce,
                    0x18dbea930f25fb6f,
                ])),
                Felt::new(BigInteger256([
                    0x875e827fdd0caf3d,
                    0x1acbb0ef98bc6c03,
                    0xc0ea68ed5cc45667,
                    0x3078e5be1ec3c6a7,
                ])),
                Felt::new(BigInteger256([
                    0x922f3bf8ebb55047,
                    0x01369de8017ef938,
                    0x47fe3ba944c65265,
                    0x2e5cfb77dbcf8c52,
                ])),
                Felt::new(BigInteger256([
                    0x09077f400fb9f92b,
                    0xa51a3f7b81687462,
                    0x8be8a1b561d95928,
                    0x12ac89236959dc90,
                ])),
                Felt::new(BigInteger256([
                    0xf3de2c617496f342,
                    0x81ac6ff0b8a5e83c,
                    0x108db6464c63fc70,
                    0x1431621ac4dd79a7,
                ])),
                Felt::new(BigInteger256([
                    0x1f6fa38fee5fde27,
                    0x01948feb400e7cd3,
                    0xcbdea35a6ad30a0b,
                    0x25bbc0795aa4bd03,
                ])),
                Felt::new(BigInteger256([
                    0xd0dbc434b25f8d6c,
                    0x298c15d8e0bc854f,
                    0x435a6ed12d9c32f9,
                    0x1899dd2b784393f7,
                ])),
                Felt::new(BigInteger256([
                    0xbc47e1b7112bcbb3,
                    0x5774780cfb15ce7d,
                    0x423111e78b86605d,
                    0x10adce3581b57bf4,
                ])),
                Felt::new(BigInteger256([
                    0x5a4dbf05f5dfdae2,
                    0xb3326d35cfd61ec6,
                    0x8c0dc00c7dcae0e7,
                    0x1c9df1d6f7bbd892,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x73a9bc53005543a3,
                    0xfe230a2db918b6d4,
                    0x16b124e9913e2e1b,
                    0x18927a16a1913367,
                ])),
                Felt::new(BigInteger256([
                    0xeaa7629492491079,
                    0x594b86faa140ffab,
                    0xfa08e78538540f54,
                    0x0faeefb6bc46435c,
                ])),
                Felt::new(BigInteger256([
                    0x142cb887bdef70fe,
                    0xeaad93e0c7ecda08,
                    0x9853833e55ce24d0,
                    0x27c98aa41cb32252,
                ])),
                Felt::new(BigInteger256([
                    0xfae68f4a1b3e1129,
                    0x22bcc12fc63aa80c,
                    0x4dfe112e6a4a45af,
                    0x25c94fd3c6468b27,
                ])),
                Felt::new(BigInteger256([
                    0x5cdf4874bf650fed,
                    0x14faabc143ed5df9,
                    0x6b31e4889821a4ff,
                    0x0096c01d4fa8bcf4,
                ])),
                Felt::new(BigInteger256([
                    0xa7d87c523ccba3b7,
                    0xe0cd28e862c83c82,
                    0x487c8e2f04dd298d,
                    0x172264c8f17793ee,
                ])),
                Felt::new(BigInteger256([
                    0xa6b9d53dbaf3764b,
                    0x9f3ad733e2c02360,
                    0xe9c6caeb25ad88b5,
                    0x28984353e55bd009,
                ])),
                Felt::new(BigInteger256([
                    0x8a9edd1c019259d7,
                    0x6434a01aba0fa027,
                    0x49dc42513d54225a,
                    0x06d899d4e14eabbb,
                ])),
                Felt::new(BigInteger256([
                    0xba5193d0b179269b,
                    0xec86c38817f80861,
                    0x0bd987675a752705,
                    0x01cefac4169dc93f,
                ])),
                Felt::new(BigInteger256([
                    0x338ef8ab14bde629,
                    0x894c615bb4b42857,
                    0x62d66ff3c53dfc8e,
                    0x2e3541e8cb65e590,
                ])),
                Felt::new(BigInteger256([
                    0x11c73d51aa5706fc,
                    0x221d1fb50321972d,
                    0xcb5f27a77055a257,
                    0x2896fee8e567ef1f,
                ])),
                Felt::new(BigInteger256([
                    0xa1b69ce554f11a78,
                    0x1d2704e2bb5e583b,
                    0xaa6b67520c34466f,
                    0x2df23777e550ab22,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xa583948ac6e31666,
                    0x09c69130e8cf3942,
                    0x25d51720d71019b9,
                    0x196af99e9019a776,
                ])),
                Felt::new(BigInteger256([
                    0x3cc52fd79d64bdc4,
                    0xf80e99ed099f9e64,
                    0x71609a3ba6f4008c,
                    0x1936a5150567ba40,
                ])),
                Felt::new(BigInteger256([
                    0xf7d105eca60a26db,
                    0x0b1e0a405563137b,
                    0xa7dd1b704227dfa8,
                    0x133a0398f2d1daa0,
                ])),
                Felt::new(BigInteger256([
                    0xa5c446b518266e1d,
                    0x30b0dc43f289265c,
                    0x9dcca9efbaa8ff4f,
                    0x0bdc0c01403fa371,
                ])),
                Felt::new(BigInteger256([
                    0xec23d5a2faaaf7a6,
                    0x00c395a77a81970a,
                    0x583e9edcec075e2f,
                    0x14b6c2a6ad3ee979,
                ])),
                Felt::new(BigInteger256([
                    0x07c2d499cc51ad34,
                    0x654220774d2751cf,
                    0x8ebe247d9cfe9699,
                    0x10fbec5cdc2e702f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc724b2dce59b45b6,
                    0x5d7ae4cd77fda426,
                    0x5bac4ea5725d08d6,
                    0x2df0fa57e577a245,
                ])),
                Felt::new(BigInteger256([
                    0x0e3a9614f36e8188,
                    0xeb306c5ea9cb8530,
                    0x4cb12da025d56483,
                    0x3dd517f523fb6a01,
                ])),
                Felt::new(BigInteger256([
                    0x40332c6263b8c687,
                    0x3e3d2e458d274d9b,
                    0x5e93a39db6268591,
                    0x2f5c29fa8b315f45,
                ])),
                Felt::new(BigInteger256([
                    0x041f9e1a80854ccb,
                    0x42fce0f517dd2e72,
                    0xc9246df4111c2707,
                    0x09db45fa039fa789,
                ])),
                Felt::new(BigInteger256([
                    0x5da87718115baed4,
                    0x2183ea109371f83e,
                    0x3fee28e80b4ea43d,
                    0x1f2e40173e3349ba,
                ])),
                Felt::new(BigInteger256([
                    0x15277862c0149d83,
                    0xdcc60c4abe811b59,
                    0x84d02c58657fcddc,
                    0x218f7b0cf98c1160,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf31f07a3899bdf5f,
                    0xa747797e6b58e8ec,
                    0x2517ec675ec23a65,
                    0x305bf5985b98d497,
                ])),
                Felt::new(BigInteger256([
                    0xfe0ad5aaa6577767,
                    0x2333aea8c166a9e6,
                    0x6e0dc42fddd13cca,
                    0x336f2ed050847909,
                ])),
                Felt::new(BigInteger256([
                    0xf6b5223dcbad12fa,
                    0x4a7e53d24b035e69,
                    0xa6ed8d2d81ca9493,
                    0x33d5c3506b00a434,
                ])),
                Felt::new(BigInteger256([
                    0x027b1e43c0cea4e3,
                    0xfb0e2a71b936a499,
                    0x019d4e54901666e5,
                    0x3cc63e6c10b2b9c3,
                ])),
                Felt::new(BigInteger256([
                    0x9a05dffaeca9156e,
                    0x9585fe0e72f35bff,
                    0x0fe86173f0190c5d,
                    0x10a5e97f53e70239,
                ])),
                Felt::new(BigInteger256([
                    0xb7d45940d9d36a57,
                    0xd5fe6e71dd418900,
                    0x3ca752913b5b87c1,
                    0x344d34c546e9be76,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x57faba040d81812f,
                    0x01a4ef046b6842bf,
                    0xfc91b8ab51e161a7,
                    0x09acb7eaaa8f57b6,
                ])),
                Felt::new(BigInteger256([
                    0xb6336ad6d686ace3,
                    0x0fddb0dc90a6c3b4,
                    0x87afb006baa053e5,
                    0x1cabedfd81f7193a,
                ])),
                Felt::new(BigInteger256([
                    0xf81b31f0b874cb3a,
                    0x78adb9c85b8ce85a,
                    0xaea6184923521e50,
                    0x3198c60045a5a889,
                ])),
                Felt::new(BigInteger256([
                    0x0e54dee06a5f2e66,
                    0xbd8807e852aa10bd,
                    0x306ae82c188d13ac,
                    0x2ee89add65dd30fc,
                ])),
                Felt::new(BigInteger256([
                    0x9f076ea88b5412a0,
                    0x8ec9310720266d6b,
                    0x2d7b8fe9baa3f00c,
                    0x2651752b4c206be5,
                ])),
                Felt::new(BigInteger256([
                    0xe1ff77f62c0c7f9e,
                    0x456cb96228e40eec,
                    0x128a16ea583c68cd,
                    0x03110d4eb9acb084,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x012e6b52313cbf8f,
                    0x1b524b9b42236655,
                    0xca1cc33687e80f55,
                    0x0c463db064f386bd,
                ])),
                Felt::new(BigInteger256([
                    0xc0104f2f8514cdec,
                    0x7c0d343c5434e7d9,
                    0xf0bcbec68e0c09a8,
                    0x29f13d48c1770c24,
                ])),
                Felt::new(BigInteger256([
                    0x7b9be909e477b5f5,
                    0x7945103c4dfae882,
                    0x7fa8dbbfb853c07d,
                    0x029618071a9bf642,
                ])),
                Felt::new(BigInteger256([
                    0x9d7b8328b98e0e18,
                    0x105ae01dcc48f349,
                    0x978a99d059f8b0cf,
                    0x26779335994029ad,
                ])),
                Felt::new(BigInteger256([
                    0x2f88bdeec46036f1,
                    0xd0a5f8bf44e77ab5,
                    0xb5b59a9d56ba01b1,
                    0x28c26e607125149d,
                ])),
                Felt::new(BigInteger256([
                    0x1a5b0af12b3ac1c6,
                    0x953df644805a8bac,
                    0xba9ddf9885863d41,
                    0x3bc3b15ece58831f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0b7ef8ed2f3140df,
                    0xdfd1a2705bb70fe6,
                    0x84a9558096337ba5,
                    0x11dd86d36c33365e,
                ])),
                Felt::new(BigInteger256([
                    0x9913cac3b13fcbc2,
                    0xd4c039c0c8250e2a,
                    0xf489ef565b2cba2f,
                    0x3ea81179f2e612bd,
                ])),
                Felt::new(BigInteger256([
                    0x4ebb1bc31d05a34f,
                    0xe15e49130e93bd83,
                    0xdf009df5e80d592f,
                    0x205e53e374c82285,
                ])),
                Felt::new(BigInteger256([
                    0xd55dbee0827ef863,
                    0xa316b220cf4b7f06,
                    0x893bc3c756c09c48,
                    0x24eb78720ec303f0,
                ])),
                Felt::new(BigInteger256([
                    0x6c8660113ee7b530,
                    0x2b8edf404736dfb2,
                    0x18ef6d12bb8ea5c6,
                    0x2488ad2581e4499c,
                ])),
                Felt::new(BigInteger256([
                    0x7017fec0c3581c13,
                    0x2d7f379a9f324212,
                    0x2afb45de13f28e5b,
                    0x01e39457d7469f5a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xffdfe10a250852cf,
                    0xed53bd908a831d3d,
                    0x6b92d7f6dd9a71ec,
                    0x1fec6ea3dd4fd094,
                ])),
                Felt::new(BigInteger256([
                    0x34db1a63d3af0b36,
                    0xbd86259cb33dd003,
                    0xdee9e9c6e80e1f9e,
                    0x16afee90261ce600,
                ])),
                Felt::new(BigInteger256([
                    0x1d43172472a72e3d,
                    0x5b7dbc6d8d420ee0,
                    0x2ef57e8bf8c34bb0,
                    0x1f2d9913478e8da5,
                ])),
                Felt::new(BigInteger256([
                    0xd5a8932cf8c0b6c9,
                    0x69f6039ac3e5f9d5,
                    0xf7bb5d66df9134da,
                    0x2518c01b687fbb4e,
                ])),
                Felt::new(BigInteger256([
                    0x5ead861f94a3bf58,
                    0xb2bcd086c0d9116d,
                    0x6b81fdc37439965b,
                    0x0b9d079b154dfb35,
                ])),
                Felt::new(BigInteger256([
                    0x36ad7e09418261fa,
                    0xa9d840be614a3057,
                    0x16bf565cea6391c6,
                    0x21fed46620f64013,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04c465b3896977d6,
                    0x3a939b5fb4c251cd,
                    0xf04b1b23eacfdd91,
                    0x242ceb5c18b64821,
                ])),
                Felt::new(BigInteger256([
                    0x8dd69e97ed9ffbf6,
                    0x8442db07a7a08a16,
                    0xac8ae4e0ebbbb777,
                    0x31ff01b6fdb0be9b,
                ])),
                Felt::new(BigInteger256([
                    0xf95ae6b891731f58,
                    0x11d277eec88cd83f,
                    0x57a00090f8f36488,
                    0x34ba4fa647384881,
                ])),
                Felt::new(BigInteger256([
                    0x55e246741351cdc6,
                    0x52ce78a2dff36a8a,
                    0x6d0142482c534cdc,
                    0x3c710c1c633c3d25,
                ])),
                Felt::new(BigInteger256([
                    0xbec13f63e1a2f3ab,
                    0x64fab190f53ce8cb,
                    0xf9f8e378e7767f77,
                    0x219606104cd3d6e4,
                ])),
                Felt::new(BigInteger256([
                    0x0e0fd9f9be72976a,
                    0x186a13c7818ed25c,
                    0x61a9e7913357f8ce,
                    0x299316e57e17e5e1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf71a8a995094addc,
                    0xeb6b3b88f8ece74a,
                    0x1700c517c26b5295,
                    0x1a694602d841ed4d,
                ])),
                Felt::new(BigInteger256([
                    0x70de179a0176a4f7,
                    0x5300eabe2fe80c66,
                    0x898832df60864790,
                    0x0c419fe5c5b10d7c,
                ])),
                Felt::new(BigInteger256([
                    0x509d5a8059995145,
                    0x7c7cca09dc9d7667,
                    0x39078448d4c3c8e3,
                    0x2c6a5296e033c3b5,
                ])),
                Felt::new(BigInteger256([
                    0xd3a8304666589687,
                    0xc3a4e45eefb2a7f9,
                    0xcedffaa330893440,
                    0x3a74d1151b01c4bb,
                ])),
                Felt::new(BigInteger256([
                    0x9d5cd8ee1a0caef7,
                    0x032c528e93d3a219,
                    0x9d9c003db2dee3a1,
                    0x353776a91aad3ec3,
                ])),
                Felt::new(BigInteger256([
                    0x91e6fc63dad7afd5,
                    0x714ae7b24c7b05cd,
                    0x6be56b02f8b05b6f,
                    0x3665524c8958ae2d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x68f1541e333b9482,
                    0x70c28d9ab0447d0b,
                    0x006f2a435ef10e82,
                    0x2aa072b825f4c8e9,
                ])),
                Felt::new(BigInteger256([
                    0xa8be117b61cb5434,
                    0xc820c0590dcf3f90,
                    0xfc2d28b03eb3314e,
                    0x13b6dfe72411d185,
                ])),
                Felt::new(BigInteger256([
                    0x7d678aaa4bd73c8b,
                    0x56cade2494846e04,
                    0x2f29f3792f60986e,
                    0x11d6d54e48434480,
                ])),
                Felt::new(BigInteger256([
                    0xb144de68c5480620,
                    0xa3fe5f37d37fc958,
                    0x9bb33ae12ede7927,
                    0x2e5f4bae0e1ffaa1,
                ])),
                Felt::new(BigInteger256([
                    0x32e5d04daa43f020,
                    0xe7999d6a61f3b5d1,
                    0xa7d818be8c3d898f,
                    0x0f9014f7122fffe2,
                ])),
                Felt::new(BigInteger256([
                    0x825c89b02d621912,
                    0x32fdb4f09e57a471,
                    0xda5a0263f7e4e662,
                    0x3786d53d6d73c04a,
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
                    0xdcc101dd5b51582c,
                    0x5e9cbb16bfeb103d,
                    0x8433ce9290ff467c,
                    0x0189745ac5f34634,
                ])),
                Felt::new(BigInteger256([
                    0xdd71d40e8712d0ce,
                    0x65f8cf3a5d157d08,
                    0x580253d750f1fdd5,
                    0x29ecc4bf9f06e180,
                ])),
                Felt::new(BigInteger256([
                    0x5076ebd9e019ed7e,
                    0x49c6f8e9cd0c6f95,
                    0x148895ca70dfe464,
                    0x0909060069dcfc25,
                ])),
                Felt::new(BigInteger256([
                    0xf31bb176c381b95b,
                    0xb99a007be9e3014a,
                    0x99280e58dc764573,
                    0x1da8b1a35cb477e5,
                ])),
                Felt::new(BigInteger256([
                    0xae5b7ba88264d551,
                    0x773e9fd0327159bc,
                    0x2ed70be8cf791a82,
                    0x36b6e28e94722c56,
                ])),
                Felt::new(BigInteger256([
                    0x5c896e07b7d1ce48,
                    0x5109c45ce2df1d6d,
                    0xd2bcc8b1f44c2530,
                    0x0b7700e58a7ccd17,
                ])),
                Felt::new(BigInteger256([
                    0xb13e3a043c7e14ec,
                    0x2fea00481531be17,
                    0x3fff5ac7589da85d,
                    0x0bfdccb2517f69a8,
                ])),
                Felt::new(BigInteger256([
                    0x5c2a0da110a7d760,
                    0xd9fbfe8d4cd70b7e,
                    0x683ce31cc9946fb2,
                    0x2a8859e21dbb642f,
                ])),
                Felt::new(BigInteger256([
                    0xc163160383a863f1,
                    0x9b6aa0e81bc963f0,
                    0x334ec95835f2a697,
                    0x372fcf36df54fa5d,
                ])),
                Felt::new(BigInteger256([
                    0x54d07805f80cbda5,
                    0x8f54486d478010fe,
                    0x95a8274ea62478cf,
                    0x281b9989a3d4d30d,
                ])),
                Felt::new(BigInteger256([
                    0xc47b5a7dcc5335bb,
                    0xbda2bb549d778a72,
                    0x8825845ab5d7a999,
                    0x0f4a952b4b02741c,
                ])),
                Felt::new(BigInteger256([
                    0xdbe4618629149a35,
                    0x0c8d7edde3fefcaa,
                    0xdb320ddfacab3ac5,
                    0x39ec35bef68c1ba4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x78553b55b0bbe9e8,
                    0xd0ff53443d2f8f57,
                    0xa3dee7e7f2c4353d,
                    0x26651d1b1cf7ceb6,
                ])),
                Felt::new(BigInteger256([
                    0xec9026aef479c491,
                    0x3ce86283327008aa,
                    0xb09f8154d15a1b4d,
                    0x15c9caa3aa8d3cdf,
                ])),
                Felt::new(BigInteger256([
                    0x52fd0fc5e5704030,
                    0x58ab52bfaf9dc274,
                    0x4207fa0b46a13db0,
                    0x39b4c14363f2bdba,
                ])),
                Felt::new(BigInteger256([
                    0xb552a1b2fa90c7fd,
                    0x1e7351d75c73097c,
                    0x06685f558a5d54b6,
                    0x01a72bc778095975,
                ])),
                Felt::new(BigInteger256([
                    0x336eac5ea1231c71,
                    0x76d7eeb7bce202ea,
                    0x49ce6e62a3fc8998,
                    0x06f81dab7d6630c6,
                ])),
                Felt::new(BigInteger256([
                    0xccf1e73449fe4997,
                    0xbe9683d35a70b649,
                    0xca805046113d886f,
                    0x0236cac269c2e2d0,
                ])),
                Felt::new(BigInteger256([
                    0x451d0cd4f777060a,
                    0x38ea547b8eeac801,
                    0xdb3ea9bf51780b44,
                    0x27a645e699bf290f,
                ])),
                Felt::new(BigInteger256([
                    0x5e7f565b17716491,
                    0xe16bd8b467b6dc0f,
                    0xb42ae6051339df60,
                    0x0b2e72891da1cae4,
                ])),
                Felt::new(BigInteger256([
                    0xe3aa01202f480e0d,
                    0xbe7f708da4460bf7,
                    0x4d770fab92eee25b,
                    0x2518e13ba5ac29a2,
                ])),
                Felt::new(BigInteger256([
                    0x63894c9485674b25,
                    0x584ca4df44d046fe,
                    0xbe9d4382f483a967,
                    0x24f2edddb2b7d09d,
                ])),
                Felt::new(BigInteger256([
                    0x241e2842001b38ae,
                    0x656f26afbffeca08,
                    0xd50a6a1fee3e0db0,
                    0x26db38be2a3ea526,
                ])),
                Felt::new(BigInteger256([
                    0xf74f77c6055dea3e,
                    0x31c2548288f08be0,
                    0x0312ce3dc1651c72,
                    0x39c767789a01d8a5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x460493c371433b08,
                    0x97ac401dc99b15a2,
                    0x511178cdaa058234,
                    0x0ccdbc4c2fbf8199,
                ])),
                Felt::new(BigInteger256([
                    0x048601f41afb7357,
                    0x6d1cca230e275af0,
                    0x66891a027b84f14c,
                    0x3a8ec8079e40a0cd,
                ])),
                Felt::new(BigInteger256([
                    0xd4b4b6699e4eeb59,
                    0x2856ff5418894591,
                    0x352a5ffb9df55040,
                    0x2d308e1daf2adfc7,
                ])),
                Felt::new(BigInteger256([
                    0xc9b731accb5ce70c,
                    0xb4350290bf14a69e,
                    0x60c1f15159569f57,
                    0x02e787a14401285f,
                ])),
                Felt::new(BigInteger256([
                    0x4f7ce0a835559e4f,
                    0xa7126e68763ea343,
                    0x96cd8224bf67649a,
                    0x02d794d790bd2ee0,
                ])),
                Felt::new(BigInteger256([
                    0xa35e04196291a6f9,
                    0xd46c37b51b005052,
                    0xb843ca20bb4b6d2e,
                    0x11a2953c77facde8,
                ])),
                Felt::new(BigInteger256([
                    0x7f1898798be71048,
                    0x8db8fa0e0b658e4e,
                    0xbc871b53d4cb365f,
                    0x04c6d5fc1fcd2659,
                ])),
                Felt::new(BigInteger256([
                    0x67cea27934b289dd,
                    0xaea6d8386afbfc28,
                    0xd34462319931e15d,
                    0x2a3875ac02b6d299,
                ])),
                Felt::new(BigInteger256([
                    0x5ea7794cd1f498e7,
                    0xd21b28b9141d92e6,
                    0x2b161c5573671036,
                    0x22a96b6b3baf8889,
                ])),
                Felt::new(BigInteger256([
                    0xe5b4a3994b7abbf3,
                    0x520bee496fed74a1,
                    0x22e17c95defd1789,
                    0x099f2753ac545263,
                ])),
                Felt::new(BigInteger256([
                    0xa8661bb9aca68129,
                    0xba2fbecefdfd3d46,
                    0x0453ac7104e11d8c,
                    0x1ce8e37446e26b77,
                ])),
                Felt::new(BigInteger256([
                    0xc92d8916641d0d84,
                    0x7986c3c02abc91a3,
                    0xc781619bc3a9fc4d,
                    0x09059e5974958858,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8ae541feef646ddf,
                    0x559ffe67f11fe042,
                    0x377ea489220c26cd,
                    0x11cf1fef0f9ce956,
                ])),
                Felt::new(BigInteger256([
                    0xbdc43f9df1563f81,
                    0x1c57272ac9f8a6d1,
                    0x832331e2e4bcd74e,
                    0x3f8f226b70123ea8,
                ])),
                Felt::new(BigInteger256([
                    0xbf130bb6f172851c,
                    0x3710937c80b6f483,
                    0xd80c5a3685efd634,
                    0x1936cb9dba8acf35,
                ])),
                Felt::new(BigInteger256([
                    0x3b2bcfdcca29098b,
                    0x3b4c9aa834a33e1f,
                    0xe7e566013ab3fba3,
                    0x260fc69aea0cd591,
                ])),
                Felt::new(BigInteger256([
                    0x95ff4eb3444a3660,
                    0x91c66b9e1269c1b0,
                    0x1fa20db52f795a3d,
                    0x0c12d6f66bf5c8f7,
                ])),
                Felt::new(BigInteger256([
                    0x4537dac24b455da0,
                    0x9e7aa6fb198a5c85,
                    0x28afdfe8b69a653b,
                    0x08403135a4a64afc,
                ])),
                Felt::new(BigInteger256([
                    0xd06879109eb255fc,
                    0xf63c485a4233ba75,
                    0xf86416609624d51a,
                    0x04a91358bb5b3c1f,
                ])),
                Felt::new(BigInteger256([
                    0xd2091a98c376e675,
                    0xebec86019b43d483,
                    0x186522a9bbd5f378,
                    0x0f4c883cffc2add9,
                ])),
                Felt::new(BigInteger256([
                    0xec3821d0c156219c,
                    0xa54c522f00b8d0f9,
                    0xa36065ca747b460a,
                    0x3a0925722b720c9f,
                ])),
                Felt::new(BigInteger256([
                    0xab1d360943a5ff65,
                    0x01ef035df3eb6e3d,
                    0xdcad199e76d74b25,
                    0x20ae1e93e5283418,
                ])),
                Felt::new(BigInteger256([
                    0x8aa21254e3563c63,
                    0x301ee8149c9bdd94,
                    0x35848b5dd2bb259e,
                    0x2e164514d5c75030,
                ])),
                Felt::new(BigInteger256([
                    0xbb9fdb61c2ca5e37,
                    0xc740219d9c4a9883,
                    0x1ee343bcb409cb66,
                    0x1c5478353631c13b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2b9c8ab17aa1cdba,
                    0x9c82456774ac9de5,
                    0x69c49eebdcb605d0,
                    0x291d4cbe3b5c1870,
                ])),
                Felt::new(BigInteger256([
                    0x6430bf448c65adb2,
                    0x0e5446f51ea6ef4e,
                    0x7986f5b5292d11e3,
                    0x29a938fcf98b94bd,
                ])),
                Felt::new(BigInteger256([
                    0x2356b56a07a0711f,
                    0x5ff32e3a2cf8281e,
                    0x9e3f2817698e9d6c,
                    0x3ad38de8ad787b1f,
                ])),
                Felt::new(BigInteger256([
                    0x832843d0bc12d705,
                    0xa7d081afe1722c34,
                    0xe4841fd81252aa01,
                    0x026072dccbfc2bc8,
                ])),
                Felt::new(BigInteger256([
                    0x47826b1cee780fc7,
                    0x8787be28a123decb,
                    0x6ae2123ff8e1392a,
                    0x03c526f5793b697e,
                ])),
                Felt::new(BigInteger256([
                    0xffd7fa5c4a5c72d5,
                    0x06bbdce203ed5bd9,
                    0xfea96b63f6728076,
                    0x0025d4b81b6164c5,
                ])),
                Felt::new(BigInteger256([
                    0x18476bb658035a36,
                    0xca86cf1a62ddd403,
                    0x0a68f055a45a8a38,
                    0x35ec55fff5ab2713,
                ])),
                Felt::new(BigInteger256([
                    0x1e48b826b80780ca,
                    0x3578dbfd2c82e8ba,
                    0x2b41f40a55b4c940,
                    0x360e5e95faca16e0,
                ])),
                Felt::new(BigInteger256([
                    0x914bbf3d59a54c1f,
                    0xf317a4e8908641ff,
                    0xd3e48cd0cce1e3d2,
                    0x0c82a92cfd96cd1f,
                ])),
                Felt::new(BigInteger256([
                    0x0421c87e35162232,
                    0x8d9019b1e0b02c8e,
                    0x032e2a7e68c96aa2,
                    0x3fda4ae40b133f2e,
                ])),
                Felt::new(BigInteger256([
                    0x1893272b6134a705,
                    0xb0288ac6ab3a65bc,
                    0xccdb687374f87f98,
                    0x177dfc5d758d7bb0,
                ])),
                Felt::new(BigInteger256([
                    0xd55ecde8c7399894,
                    0x9637047205a50c75,
                    0xa01cbda9fdcf1f11,
                    0x344ec82eb806fe48,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa42da70a2f4d390d,
                    0x9a27e026ee6576ba,
                    0xebbaa51829306944,
                    0x2797ba0d06d1ab81,
                ])),
                Felt::new(BigInteger256([
                    0x73abaa781f7696cb,
                    0xbaab4a5fd14691a1,
                    0xec5907fb5d71d0e8,
                    0x19c9df8be4a3ad1e,
                ])),
                Felt::new(BigInteger256([
                    0xdd6f840a3b521ab9,
                    0x7b4f201f9fc38450,
                    0x1edd6ef30e0ce9ab,
                    0x1efb045910be463f,
                ])),
                Felt::new(BigInteger256([
                    0xb2ce45a7c637e3f0,
                    0x91114740f09e500b,
                    0xeec74f77dfa81a12,
                    0x0809c4e6e5230918,
                ])),
                Felt::new(BigInteger256([
                    0x5be5b77ccb2f76f6,
                    0xfbf5bb7e67d98e19,
                    0x1cec771264e27a3b,
                    0x3436449b34302f6c,
                ])),
                Felt::new(BigInteger256([
                    0x751a68eb128b7973,
                    0x8230f63591d6d83c,
                    0x89d13b68ad8c9e92,
                    0x25cfcf33c0945f1f,
                ])),
                Felt::new(BigInteger256([
                    0x916c0f916bf3cc3e,
                    0x081e5d9722375202,
                    0x4ff48698186bfd41,
                    0x175d313be18e4e57,
                ])),
                Felt::new(BigInteger256([
                    0xac26d423d16e641e,
                    0x464c4f53f455fce7,
                    0xc2706d4556632e54,
                    0x23294b6227f0da90,
                ])),
                Felt::new(BigInteger256([
                    0x08e2142391fc833f,
                    0x0ba9abc4f84a890f,
                    0x799e8f6678dc13fa,
                    0x0071688b5c57f8f1,
                ])),
                Felt::new(BigInteger256([
                    0x96c6cb092744f8aa,
                    0xc02eac9ae38a44e1,
                    0x954811afac0c0d4d,
                    0x1b8418b7fe48fe3c,
                ])),
                Felt::new(BigInteger256([
                    0xb51db9e4d0f70aaf,
                    0xe115ee1728e31697,
                    0x278691b040c987ec,
                    0x053cb1c70879269e,
                ])),
                Felt::new(BigInteger256([
                    0xd32f07b2c649008d,
                    0xbe99dbe73d75fb63,
                    0x7b8cb88ab9cb9e60,
                    0x31d891d929a3993c,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xe77dd01fe9750dfb,
                0x81632ec4f86f517b,
                0xc3dc3a1703daee05,
                0x376a5d5152003971,
            ]))],
            [Felt::new(BigInteger256([
                0xe7ad41868eb826e4,
                0x615b8bcdfc02be62,
                0x94d3e317d0438c0c,
                0x25bb3d65d0036e30,
            ]))],
            [Felt::new(BigInteger256([
                0x0b18aa8782eb8e64,
                0xf271aefb5adbd761,
                0x8840401e79e906c7,
                0x195a4469c2a16c47,
            ]))],
            [Felt::new(BigInteger256([
                0x7d174608be3cb9ee,
                0xd7611a02e0272a29,
                0xa3580ffb5b414062,
                0x303c893fddd666e0,
            ]))],
            [Felt::new(BigInteger256([
                0xcb0e4f309f6d65f3,
                0x0da3d0a34431b27f,
                0x07c038cedae6c0bd,
                0x1b78c3d0e659f76a,
            ]))],
            [Felt::new(BigInteger256([
                0x6f732b31258ed379,
                0x9426a86fb891c330,
                0xcb6ee41886c1f3d2,
                0x38b29be069c35d54,
            ]))],
            [Felt::new(BigInteger256([
                0x187e3bcee4a9f859,
                0xbafd1b202754bb45,
                0xfdb07cb286573b24,
                0x21e512f04a092cd4,
            ]))],
            [Felt::new(BigInteger256([
                0x72be58a9ce249d71,
                0x01b2c64904f335c1,
                0xe07f2ec550eb83b2,
                0x0f88aebe9f38ec64,
            ]))],
            [Felt::new(BigInteger256([
                0x63c6bda707848fa1,
                0x4a02c2f9e800f086,
                0x5c6f825218fb78a8,
                0x0f6d59023d5400f5,
            ]))],
            [Felt::new(BigInteger256([
                0xe01b383b2690ba2f,
                0x5e4e118d890c2223,
                0x1cfacd063f6315c6,
                0x1871fede91704136,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 12));
        }
    }
}
