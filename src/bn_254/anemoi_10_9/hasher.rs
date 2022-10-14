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
                0x27d8531ac97b9e33,
                0x3e1521858f24832a,
                0xda29c7a828f706e0,
                0x28aeb07208bf6e5d,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x1e8cb69850ffc830,
                    0x53770dbcfeeed5f5,
                    0x057074d8d291d0ee,
                    0x2b6805d63295c8f3,
                ])),
                Felt::new(BigInteger256([
                    0x82781087e40f1230,
                    0x2ad82a3f239d06f8,
                    0x2fab870d7097bb63,
                    0x01c11b189f593f44,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x571b6733b61f0807,
                    0xc3ee0be3d3c5a4c8,
                    0x5da9bd6362232a8b,
                    0x28b03d59dcd9ec54,
                ])),
                Felt::new(BigInteger256([
                    0x99e498399aa77c8b,
                    0x56349abeb92b1dff,
                    0x26d34733f45c33cf,
                    0x1aae662b95b751d2,
                ])),
                Felt::new(BigInteger256([
                    0xfb9a8591aa6fb445,
                    0x847e23532fcc75f2,
                    0xf4ad237cce33a9dd,
                    0x0a89ecd51d00f18c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x455c54065baecf8a,
                    0xe3150794d3f499c4,
                    0x49ba6a61016869b3,
                    0x0e35dfd095a0a4ff,
                ])),
                Felt::new(BigInteger256([
                    0x214414fc8af067cd,
                    0xfa95c2712ef5a1d9,
                    0x542b9907d559e89d,
                    0x14c95350aa522574,
                ])),
                Felt::new(BigInteger256([
                    0xae3a8be175fee963,
                    0xcf5feec1064305e5,
                    0xc98b1d7eb1ec2c4f,
                    0x05c715a8d1086470,
                ])),
                Felt::new(BigInteger256([
                    0xd3276d80a8179681,
                    0xc06a6fce0e648b10,
                    0xc218bb22e5796173,
                    0x27f690ca5a7c6b5b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc649c2880f3ff30a,
                    0xb3505e6dd11c4df6,
                    0xa1de9ad660eb8a71,
                    0x094547d31040e624,
                ])),
                Felt::new(BigInteger256([
                    0x83b6a56164186d32,
                    0x1c4e80b0a39f1169,
                    0x8763c115aff70ef3,
                    0x1ebc0657dc6d2edc,
                ])),
                Felt::new(BigInteger256([
                    0x987bf3ab2b8ea091,
                    0x13ad5e5543e9bc64,
                    0xe528568bcce759ef,
                    0x272ca17c293eaea3,
                ])),
                Felt::new(BigInteger256([
                    0xde36dc45a3e3641b,
                    0xa9f7465fdc2a6282,
                    0xe737f4de34bf4a68,
                    0x246c32ddb7824898,
                ])),
                Felt::new(BigInteger256([
                    0xb83eb0cee423a81b,
                    0x930a67b4858a6ede,
                    0x23e6b65a7362b902,
                    0x1555dd7b485a9ef1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5f2701ad3b362b84,
                    0x5261353e5b660af2,
                    0xdba77443b9561a15,
                    0x0a29737c2e5a3814,
                ])),
                Felt::new(BigInteger256([
                    0x36261009d9466253,
                    0x4ad7a7e92d3adcf8,
                    0x7b85067e14a5497e,
                    0x1f2af7b386b46775,
                ])),
                Felt::new(BigInteger256([
                    0x2f34ddc78f7550ba,
                    0x4719a302db497c06,
                    0xef6e8c6b033b6f35,
                    0x05b8a44b26d13fde,
                ])),
                Felt::new(BigInteger256([
                    0x4900b15693d4fbd8,
                    0x24e3851d30b9953b,
                    0x41c5fa92ab6765bf,
                    0x05adfa689fc62715,
                ])),
                Felt::new(BigInteger256([
                    0x56afe0a9d47f198a,
                    0x0c55504c450b5356,
                    0x6d8a385a0b828670,
                    0x293f190a4f1523fd,
                ])),
                Felt::new(BigInteger256([
                    0x9695f99ddeb82cb5,
                    0xb767b1ac72aa119e,
                    0x258b4931e7c7de06,
                    0x22aad6c5a7b19dcc,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x1c220c9553bf7361,
                0xc63806eb915a63df,
                0xc593a8175270c259,
                0x01c4e98489e99bf9,
            ]))],
            [Felt::new(BigInteger256([
                0x6b2068a02143e9d5,
                0x07a85ffa037864c7,
                0x3c169838f11e02f4,
                0x228a89d2455a2901,
            ]))],
            [Felt::new(BigInteger256([
                0x01deed1bbfd38870,
                0x28558c0f341e02f6,
                0x0f01cb427e1e2461,
                0x0e9b5fbad3dd0796,
            ]))],
            [Felt::new(BigInteger256([
                0xf498f2a91436b93b,
                0xe7550e55a9011709,
                0x72269384ce3e5d61,
                0x0c70d84b21d4bc50,
            ]))],
            [Felt::new(BigInteger256([
                0x6c7b38d7c780ade2,
                0xa6f87a0d1023293d,
                0xd54d05be31424ffa,
                0x0403f698a8d9d31c,
            ]))],
            [Felt::new(BigInteger256([
                0xc64d4a86a6245514,
                0x31ce8208cddb93b7,
                0x53b452837181aafe,
                0x021a5b01bdabc9a0,
            ]))],
            [Felt::new(BigInteger256([
                0xd08846ef6b542aa5,
                0x75f64f50e500af72,
                0x7a6530357ef71258,
                0x290a8750f39ace17,
            ]))],
            [Felt::new(BigInteger256([
                0x797ff664b9cd14c1,
                0x244d9e54e8347f79,
                0x05419d9f554a7e01,
                0x16059edf207036bb,
            ]))],
            [Felt::new(BigInteger256([
                0x8bf2dbedc2fb8335,
                0xc4bb1859ef1aee1c,
                0x43c246b87162eb33,
                0x0657f8bbcd4c22fc,
            ]))],
            [Felt::new(BigInteger256([
                0x5791cefb2c6e3bd7,
                0xdde9d7605a4ecfe9,
                0x0b1f556621646693,
                0x06b6b9ed0ddc7860,
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
                    0x0816571b3da2ce2c,
                    0x10a61379683aca5a,
                    0x5cde319a274caef8,
                    0x248f90678e011c78,
                ])),
                Felt::new(BigInteger256([
                    0x910fe8dcfda88dff,
                    0x0af6734e10a458ce,
                    0x825c548b244bf7d5,
                    0x23f3ef0c48337ece,
                ])),
                Felt::new(BigInteger256([
                    0x007783cd16a61831,
                    0x2709ef1f1e8de906,
                    0xc8a45d29ec6a909a,
                    0x01862847bf9fffcc,
                ])),
                Felt::new(BigInteger256([
                    0x0ce8f19a6931d614,
                    0x741d30f0354723e8,
                    0x45a959dd44b61a6d,
                    0x0254d9bd12a53997,
                ])),
                Felt::new(BigInteger256([
                    0x08bbd092fd01aaff,
                    0xc7f4d7528c517f36,
                    0x526bc1880a71d21a,
                    0x0747566e363d88ff,
                ])),
                Felt::new(BigInteger256([
                    0x63a6ea4bad2df238,
                    0x2356668adbcaacac,
                    0x8f511ba2dc349386,
                    0x078623b7ac091907,
                ])),
                Felt::new(BigInteger256([
                    0x4e4fa3b6a393e9b3,
                    0x7b9e68a4fe678350,
                    0x4dde94db2405cbf8,
                    0x1b7e7c32162d0a68,
                ])),
                Felt::new(BigInteger256([
                    0xa71adf2feac57a88,
                    0xe0d76cd5f8381bad,
                    0x88d048e81884a425,
                    0x08be53bf404c3d7b,
                ])),
                Felt::new(BigInteger256([
                    0x9d409c97c261594e,
                    0x39086e52173112b2,
                    0x3b6b3a48066f6810,
                    0x26dc92875a6f40c8,
                ])),
                Felt::new(BigInteger256([
                    0x12121d7cdf80cd6b,
                    0x0e5e74d8b3b8253d,
                    0x2242543695c3249e,
                    0x1822e0df3418e172,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x615b5c3d17a049c6,
                    0x7214286765bf0dbb,
                    0x008729bb8b076b2f,
                    0x2d5a235e05070b4b,
                ])),
                Felt::new(BigInteger256([
                    0x173250ac063800a6,
                    0x094921e591ddab0e,
                    0x480038a1c60e091a,
                    0x181946ff3962f498,
                ])),
                Felt::new(BigInteger256([
                    0xa260b17affb14454,
                    0x37c7617dfa70e4f3,
                    0x02a2fabf61c98f5c,
                    0x1602b881460b961c,
                ])),
                Felt::new(BigInteger256([
                    0x42f33ad9e0486284,
                    0x4fc90bd0722a2e83,
                    0x10329ce51c2f8c3c,
                    0x264565dc879fa746,
                ])),
                Felt::new(BigInteger256([
                    0x506577bc4fac19ad,
                    0x9d9e787c60064be2,
                    0x416e1a26bf62a66b,
                    0x1e174904affc168b,
                ])),
                Felt::new(BigInteger256([
                    0xce1e98044f66a66c,
                    0xf75b4f9a468ff9e0,
                    0x363680e7e49dff33,
                    0x1c7ecca0dfd1d6be,
                ])),
                Felt::new(BigInteger256([
                    0x20181bd509832e6e,
                    0xa291893a81ced5fc,
                    0x77026fbed3e71fd1,
                    0x21621cae4524dd3f,
                ])),
                Felt::new(BigInteger256([
                    0x547d65147f1211cd,
                    0x601754d8c6967e76,
                    0xa44e84cddf390f3c,
                    0x095837ba95063c48,
                ])),
                Felt::new(BigInteger256([
                    0x98db02c3e271ae1f,
                    0xeffaa9261f808abe,
                    0x61a9cde3c42f9781,
                    0x21d9b6fb9520500b,
                ])),
                Felt::new(BigInteger256([
                    0xa4865788c534badc,
                    0xa558474961d8c295,
                    0x08805d852bf8d82b,
                    0x196c21230c5541e5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3659111514fe44b0,
                    0x7c23f1fc95f2ea27,
                    0x5d05351e5a810c21,
                    0x1986cd35031cffe1,
                ])),
                Felt::new(BigInteger256([
                    0x3ccda39a8e170887,
                    0xa339008ece64b6f8,
                    0x9730d8c16b71c8a6,
                    0x2b44a7d1aa4dac5f,
                ])),
                Felt::new(BigInteger256([
                    0xc6be746c82eebde5,
                    0xc2dfc952b5a76cb2,
                    0x0e656c7e0c9596a3,
                    0x11d132958ee17959,
                ])),
                Felt::new(BigInteger256([
                    0xd6636a87a6074af6,
                    0xcbf611ae4dea2947,
                    0xeb070579066733ec,
                    0x003ab7c1ad0edb90,
                ])),
                Felt::new(BigInteger256([
                    0x05a315a81e478367,
                    0x17a3b1fd1afd1915,
                    0x9d29b32366e096e1,
                    0x0a394a8ea2e28409,
                ])),
                Felt::new(BigInteger256([
                    0x89aa34871434ab96,
                    0xa4d6283787f3cfad,
                    0x790b7e810f48d45a,
                    0x0d1f66dc69757b89,
                ])),
                Felt::new(BigInteger256([
                    0xdbd6e416c7e0c5b1,
                    0xcdb11903d6e546ab,
                    0x3d1ac7c3cc44fc29,
                    0x1cafc56fc32c0aed,
                ])),
                Felt::new(BigInteger256([
                    0x5f40e4b77d72739f,
                    0x6d483318a59d8935,
                    0x4bbbdddca6d23103,
                    0x2db86da63d7aa3ae,
                ])),
                Felt::new(BigInteger256([
                    0xbf35c3cf868671bd,
                    0x838b579dd8e777f1,
                    0x136f69ace8f0122f,
                    0x0ad8f1aa5fa75769,
                ])),
                Felt::new(BigInteger256([
                    0x631c2d1e2b9f5180,
                    0x7cb712fc8dc2ff00,
                    0x4f990baf468e8a75,
                    0x2c41efba962bab79,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x41cbab6e9d236a2a,
                    0xff58bbe89243ca56,
                    0x69e9b1ba73a52168,
                    0x139c678f5a869604,
                ])),
                Felt::new(BigInteger256([
                    0xc9f5f0dd8dcf1148,
                    0x97431baa1f80fab4,
                    0x049da6b8a1218c78,
                    0x1653ac415b03ddbd,
                ])),
                Felt::new(BigInteger256([
                    0x6bb55ec19cad1451,
                    0x2c43a164e614a460,
                    0xff3c703065c67805,
                    0x06df588d6083bef5,
                ])),
                Felt::new(BigInteger256([
                    0x99268615f21c86f5,
                    0xb31ae98251aaaa96,
                    0x3c903f19cf420d71,
                    0x2dbfc61cecf2fa06,
                ])),
                Felt::new(BigInteger256([
                    0x9e5833464a8609d7,
                    0x17c77f873f3d7813,
                    0x8686aad982067983,
                    0x12f6823f3a7ab987,
                ])),
                Felt::new(BigInteger256([
                    0x8347914fd6898beb,
                    0x6f1f6f970384c3b4,
                    0x31bccc754aa52288,
                    0x237eadc682e1b22c,
                ])),
                Felt::new(BigInteger256([
                    0x8b4a9cc049a34c55,
                    0x36da3a9b9d1e3a2b,
                    0x5dcda2f7e88338ec,
                    0x122b149ab0a6fac9,
                ])),
                Felt::new(BigInteger256([
                    0x92440c659142d454,
                    0x1746763f85b54385,
                    0xb619409e5bb305b9,
                    0x183beefc287f5c74,
                ])),
                Felt::new(BigInteger256([
                    0x39216d1579cfbbdb,
                    0xfa22e6b46ae4d5a8,
                    0xfe19a6b961c4e637,
                    0x24691471a45e0614,
                ])),
                Felt::new(BigInteger256([
                    0x894cf9104cdfe55d,
                    0x2621d38d01021795,
                    0x2c8868e006ea02a6,
                    0x1da1f8a749320a5f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7a9dae3d439d0089,
                    0x5ccabe0f5d088369,
                    0x4f1bd506e34add46,
                    0x269a5f23d90e8a9b,
                ])),
                Felt::new(BigInteger256([
                    0xe8c8440d8a8156c1,
                    0x6fff51e6a2f302c8,
                    0x3c5038976b76ffcf,
                    0x2eb4cca01ad2a514,
                ])),
                Felt::new(BigInteger256([
                    0xefa312f6482f2e00,
                    0xfeac9ccd5493e0ea,
                    0x59a8380655d4c1b3,
                    0x1d49f668922f74cd,
                ])),
                Felt::new(BigInteger256([
                    0x7db017692fafbe3b,
                    0x60f4226a6b2964fb,
                    0x6c87d402dad07513,
                    0x1a3c2269951bd5e1,
                ])),
                Felt::new(BigInteger256([
                    0xc6d50821d10c1675,
                    0xc734458d011776c0,
                    0xe43b3a725a988f13,
                    0x00daf880fc3d8259,
                ])),
                Felt::new(BigInteger256([
                    0xd3ab1a96b53e2eb7,
                    0xf5e4f2e8948d3468,
                    0x41226ed7463f0a7b,
                    0x1f266239c520270b,
                ])),
                Felt::new(BigInteger256([
                    0xbb4283e48d7b1432,
                    0xd48ae67b8548172f,
                    0x146b3b134f96fed8,
                    0x0160e27760459b8e,
                ])),
                Felt::new(BigInteger256([
                    0x501cc64cb2c86278,
                    0x72ceb42cadf46957,
                    0xa125362ec9239102,
                    0x116ee44002afdd9f,
                ])),
                Felt::new(BigInteger256([
                    0x9960f6212719be7a,
                    0xae775f169417130a,
                    0x7253e1bcc14f83be,
                    0x020889d6d031dc22,
                ])),
                Felt::new(BigInteger256([
                    0xb113a20802579fd0,
                    0x646d6b7188efee66,
                    0x2335fc9ddb2b40c9,
                    0x2a73c1d7b610788e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7673b30876651b2e,
                    0xbdcfab8588897d31,
                    0x037724d667140ce4,
                    0x282ac4f5cf9b3ead,
                ])),
                Felt::new(BigInteger256([
                    0x2a20dbcaaac81bbe,
                    0x4945c8ebbba12607,
                    0x9cf6fc31776b6a6f,
                    0x0140cc66d5564559,
                ])),
                Felt::new(BigInteger256([
                    0xda2e82b0e9fa3053,
                    0x3b31a14d7a696691,
                    0x28ae797f88eca69f,
                    0x15684ab2fd94d64e,
                ])),
                Felt::new(BigInteger256([
                    0x601a510b7c63a8e2,
                    0x7a01326a8391b683,
                    0xc741621932091469,
                    0x2dadea703d86e365,
                ])),
                Felt::new(BigInteger256([
                    0x20fa4f5374d91907,
                    0xa0e1fc17541c8727,
                    0x9be96ada06250ba0,
                    0x1fb3ff1334944050,
                ])),
                Felt::new(BigInteger256([
                    0x16169622eba8ac6e,
                    0x846eab5f7a97b1c7,
                    0xfa9326f5c75f8d5a,
                    0x1f543083bf4dc926,
                ])),
                Felt::new(BigInteger256([
                    0x83c397941e6a3319,
                    0xa35047c00f6adf27,
                    0xd15952c7077fda73,
                    0x0e8d741d186d64de,
                ])),
                Felt::new(BigInteger256([
                    0x7531df82d886b190,
                    0xece2c4dc1c4bdbf7,
                    0xe45f40f5c5f11319,
                    0x1ce2a894f9cca7b8,
                ])),
                Felt::new(BigInteger256([
                    0x394b83f48312a5c8,
                    0xffd1a539639198b3,
                    0x82b0a57c22fbc4ac,
                    0x2aa972ec415950a6,
                ])),
                Felt::new(BigInteger256([
                    0x4e832ea2e4c9a637,
                    0x4df23659120716fb,
                    0x3708f6b078320c1f,
                    0x1a83a2d65607dbad,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x7a60860d9e59674c,
                    0x78589e19332a108a,
                    0x24daa1fa2f47f56f,
                    0x00f41f84182a16f8,
                ])),
                Felt::new(BigInteger256([
                    0xb02fbfb0bff85941,
                    0xa4583a9e977c5316,
                    0xa3625643049bc273,
                    0x0e392dfce1b415e0,
                ])),
                Felt::new(BigInteger256([
                    0x044eba7be175e295,
                    0x6bb4f11a967a6d15,
                    0x97d63edd68446833,
                    0x0036256ac3ce87e0,
                ])),
                Felt::new(BigInteger256([
                    0x85af8a78245edffb,
                    0x830673d14e82a1a6,
                    0x9c69c4cf0ea9a643,
                    0x2689bf1fb4668453,
                ])),
                Felt::new(BigInteger256([
                    0xa9d442241b25487e,
                    0x32b74848b0ceecf3,
                    0x59e01f8a4af30f39,
                    0x2a843e6b3b426c6f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3817e925f1c8c958,
                    0xfb57669612a64ffa,
                    0x1f680e494b8f785e,
                    0x18657e6897a1bcc6,
                ])),
                Felt::new(BigInteger256([
                    0xbd9906ed897b7004,
                    0x93e749907b5ab17e,
                    0x07df5fc2c7e6e60d,
                    0x13f0648c5032d1c7,
                ])),
                Felt::new(BigInteger256([
                    0x74bd36801176362b,
                    0x0bc8a6817003e768,
                    0x76e2c8d2757ac3a8,
                    0x0f836317df96e8bd,
                ])),
                Felt::new(BigInteger256([
                    0x5f3c073b700bf19e,
                    0x553a9d680eaddc96,
                    0xa6874bf2e2820ab5,
                    0x00b7c919400e6c10,
                ])),
                Felt::new(BigInteger256([
                    0x6fd6e5e793f496db,
                    0x501dca6ecb96a8e2,
                    0xa99c4fa0458b5ba4,
                    0x27a1c91f87e4c562,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xce1d6dc338fe8d45,
                    0x4d74763137480638,
                    0x395a3d0ce6014901,
                    0x18d704a3ff749890,
                ])),
                Felt::new(BigInteger256([
                    0x86e9ed011f4ab7cb,
                    0xa99fa35f9ba41fd7,
                    0x4bb0e5edf40f82d1,
                    0x05f273d7afa586c8,
                ])),
                Felt::new(BigInteger256([
                    0x794937b24663977e,
                    0x64eedb26ef47a206,
                    0x1310eda4ea7e35a0,
                    0x0a8279b7c39e6f36,
                ])),
                Felt::new(BigInteger256([
                    0x33a6a7cd24924e81,
                    0x5aef6ff8a9533a9a,
                    0x5049d428306a456a,
                    0x1239eda48a0a056c,
                ])),
                Felt::new(BigInteger256([
                    0x4e558685f6039b9f,
                    0x21cf8fd92a3531ce,
                    0x577a15a8545a7145,
                    0x0e70c213fac45919,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0c03e5c4574036ee,
                    0x265766fb56908f25,
                    0x50f97fa76b121b30,
                    0x27a3e49613327cf8,
                ])),
                Felt::new(BigInteger256([
                    0xbbda7b6c191bd326,
                    0x71dda7b73b64d7ab,
                    0x8075b02f9ffb1fa1,
                    0x15036070037cc4f7,
                ])),
                Felt::new(BigInteger256([
                    0xe9fdbc8a6bf54a52,
                    0xee4041d4de593455,
                    0xd56d89ac5c2c06b9,
                    0x294a436023acc4ad,
                ])),
                Felt::new(BigInteger256([
                    0x8c06096f911eac42,
                    0x8c369d4a9cabad6e,
                    0x6bf0de2cdce552dc,
                    0x286a30e08e59463f,
                ])),
                Felt::new(BigInteger256([
                    0x09d11c1be43ad78d,
                    0xab1b57c648f04765,
                    0xf4e34d907536d3e5,
                    0x0274203b526f0b3f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x20d4490d2abab87e,
                    0x82b66c468eb030c3,
                    0xc5fdf6571469b92e,
                    0x22f068d4609a341f,
                ])),
                Felt::new(BigInteger256([
                    0x380694da08d36395,
                    0xa98258da7747c4f0,
                    0xffe85bcc52edfe46,
                    0x0482157c0b53c178,
                ])),
                Felt::new(BigInteger256([
                    0x85e3c72ef5ed5260,
                    0x66563dcce9a2e3a4,
                    0x9eec1bc171974662,
                    0x2157ce0db291db2c,
                ])),
                Felt::new(BigInteger256([
                    0x88b70cc158448451,
                    0xeaa30ae7100d7f6b,
                    0xd2603880c5186ee0,
                    0x1d8871b6dd0a8714,
                ])),
                Felt::new(BigInteger256([
                    0x2e10d7c45d177dbc,
                    0x0cb16488074bfa0f,
                    0xf7b0ec7055d676e3,
                    0x19a0b15b704db3b5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa4ab27264700aeb1,
                    0x57780f0775e02df6,
                    0x5321cd9c471c4139,
                    0x0e4b884d4a6c343a,
                ])),
                Felt::new(BigInteger256([
                    0x400782970c2702bf,
                    0x2a5459cd2d8b91fc,
                    0xb263c08793372455,
                    0x0777cc825bf811b5,
                ])),
                Felt::new(BigInteger256([
                    0x226756ead5d36a06,
                    0x9bdfd1454a844ada,
                    0x5504870e238b6ee2,
                    0x17b818270fc9e11a,
                ])),
                Felt::new(BigInteger256([
                    0x7978765b684c21d2,
                    0x18084354c579d421,
                    0x029f5fbdad1ae57c,
                    0x0d4098bb41ecdbf7,
                ])),
                Felt::new(BigInteger256([
                    0x7a1bf160651e9cae,
                    0xd95fef3a736f3874,
                    0x5e06475bfdcb8391,
                    0x13c79fab137967bb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x103e8dc8b6c2d457,
                    0x7be480b0f1866632,
                    0x431af6a437fcae58,
                    0x16a9cb078f098589,
                ])),
                Felt::new(BigInteger256([
                    0xe448272f47fc4c72,
                    0x2c8535ac687fa4bc,
                    0xba3d0e262ab80ad1,
                    0x0ccc7564fd69f32a,
                ])),
                Felt::new(BigInteger256([
                    0x3d7fbbb12180c382,
                    0x8b7deaf8041b2b61,
                    0x80f2e9ddc9ad1a10,
                    0x0cf0a1f8b431bd84,
                ])),
                Felt::new(BigInteger256([
                    0xe1ab7364ed59a691,
                    0xc2eb4281a9e5c3a6,
                    0x3d85ba3ceb2a66af,
                    0x06a71f2d17addcc5,
                ])),
                Felt::new(BigInteger256([
                    0x5f6d5a4c5c946cdd,
                    0x167afaf40ce3bbff,
                    0x0283e2da2efe097c,
                    0x2dd894b2a7acbae3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1c6071689a246e6e,
                    0x69c575a3daed5014,
                    0xca63765ea8f7e330,
                    0x058e6b110ff71746,
                ])),
                Felt::new(BigInteger256([
                    0x0b1b085739a2c4d9,
                    0xd84d48fa607451d6,
                    0x9583303c46dca339,
                    0x04e09abcea0beafc,
                ])),
                Felt::new(BigInteger256([
                    0x5c868835560a9ee3,
                    0xffc6d18a22a5c21f,
                    0x0be8d8ecf57bca0a,
                    0x26a665d856cc132e,
                ])),
                Felt::new(BigInteger256([
                    0xcedf79a3725c17c8,
                    0x405f88a699452c80,
                    0x5e29693ecc21885e,
                    0x29add2cf809af29c,
                ])),
                Felt::new(BigInteger256([
                    0x21572751866d9138,
                    0x7be2e412c840c66e,
                    0x403f9b59c87e25b3,
                    0x1deb9ddb467a85a2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc7bb1a8ea4077b23,
                    0x75f1648fb4a79302,
                    0x7cef93638f07a3ea,
                    0x03524fac3e1fbe74,
                ])),
                Felt::new(BigInteger256([
                    0xc3431fb4513cc211,
                    0xfc915542f1c02133,
                    0x23170f28e8506b81,
                    0x1cd7eca3e18381cf,
                ])),
                Felt::new(BigInteger256([
                    0x052e434d72676dc3,
                    0x2e43176a03097095,
                    0xe9f526012eeaf0c3,
                    0x12bee07a333e73c8,
                ])),
                Felt::new(BigInteger256([
                    0x31bda7f47031fda5,
                    0x6c942ab427d4a2de,
                    0x02fbc7a6db7a451d,
                    0x17edd39ebdd25457,
                ])),
                Felt::new(BigInteger256([
                    0xcce0d37d33ae02a3,
                    0x3d5f0ad6caa3a658,
                    0x3520655c9712e26a,
                    0x298c49f63b674450,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x979adbbd3089f281,
                    0x8897625bc7274279,
                    0xce7891c2caed8f04,
                    0x186732fe3178d08e,
                ])),
                Felt::new(BigInteger256([
                    0xbed4e3bb98a1d44d,
                    0x653d64d2301860ab,
                    0xaa776e32864717d0,
                    0x03a75a15d62d4788,
                ])),
                Felt::new(BigInteger256([
                    0x53d72724f30056b6,
                    0x076fbe48b3a0e2a8,
                    0x025e1950faeac301,
                    0x28248bc73fad7c91,
                ])),
                Felt::new(BigInteger256([
                    0x0f8ebf4d44748bfd,
                    0xda622a3ece3545f6,
                    0xb4e2982fcf4df371,
                    0x2da7f172eaf1ff95,
                ])),
                Felt::new(BigInteger256([
                    0xc0cfea8c451d44d3,
                    0xbe1103b85276080e,
                    0x2a855e7831c84945,
                    0x165684aa016e3bc6,
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
                    0x1bcd44a2bf265831,
                    0x424ddb45e0049261,
                    0xf17f31c0e3fd6491,
                    0x06e0acaeba341e7b,
                ])),
                Felt::new(BigInteger256([
                    0x5d57bc128cf191fd,
                    0x99b51130f6a1897b,
                    0x619ba1af08fb3752,
                    0x0e51f707d3618d4f,
                ])),
                Felt::new(BigInteger256([
                    0x11f0488326707122,
                    0x71c84a03c26babe4,
                    0x0b32ff183685943c,
                    0x08b1ac9a52f10e2c,
                ])),
                Felt::new(BigInteger256([
                    0x7cfdb2c4190c08fc,
                    0xca38d46ec74822f8,
                    0x1719b0e739a90f6d,
                    0x22191d6ee73d5e02,
                ])),
                Felt::new(BigInteger256([
                    0xa55c6a1d4a301f99,
                    0xc5038e53d53e8d49,
                    0x62f07d9a9db4632c,
                    0x127f62f30c637291,
                ])),
                Felt::new(BigInteger256([
                    0xefebfa9edafee8bb,
                    0x07bdd88912b6da25,
                    0x4cc1de059e3467bf,
                    0x027d0fad1b8724e4,
                ])),
                Felt::new(BigInteger256([
                    0x029558384f89dfc1,
                    0x953aa743f9ae2b06,
                    0xdbed0d2178fc6995,
                    0x03618835c66e3209,
                ])),
                Felt::new(BigInteger256([
                    0x5b97ac56562d7f85,
                    0x5b2b8add80ccd937,
                    0x24ea443dbfab432e,
                    0x0e2dfb5c64456e02,
                ])),
                Felt::new(BigInteger256([
                    0x32c6f406cbc71d79,
                    0x85e0d5b19ddac53a,
                    0x7f15cd517eb3bc08,
                    0x23ef3b99c12ecfc0,
                ])),
                Felt::new(BigInteger256([
                    0xb48088f6dbabe14b,
                    0x9547b19ec624c511,
                    0xc62581d71a72a271,
                    0x21b9c8f9cc47ee2b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9590f5fbf17ad8fc,
                    0x099c885d55a318e3,
                    0xcd51fbb8db8e1282,
                    0x30177593f389b8a0,
                ])),
                Felt::new(BigInteger256([
                    0xc47992ddab367401,
                    0x0981fa0fefa2f76e,
                    0xc83e6039966fca68,
                    0x17c27ab8f885b411,
                ])),
                Felt::new(BigInteger256([
                    0xecfe6376a74c59fc,
                    0xcbb73a3a32cc6c32,
                    0x60e0e465b8795aae,
                    0x0b82c8cdf96651bd,
                ])),
                Felt::new(BigInteger256([
                    0x4f8d463490592997,
                    0x3ab63210d5d63bb7,
                    0x3351024ca2c72668,
                    0x1a410d3d3666dc31,
                ])),
                Felt::new(BigInteger256([
                    0xb22b82b3053e4945,
                    0x4b54e2746a1c97f2,
                    0x287f75846baf4bac,
                    0x27e6616facdcd3b2,
                ])),
                Felt::new(BigInteger256([
                    0x3f4a40df5c32edd0,
                    0x3c677d80efe18a16,
                    0x30b86ad70329e81e,
                    0x2d63dffb5101ab4a,
                ])),
                Felt::new(BigInteger256([
                    0x23640894268de0e5,
                    0x6a798ea0e25a0d34,
                    0x4a943d415043d003,
                    0x01c713bfa30a7f30,
                ])),
                Felt::new(BigInteger256([
                    0xc64ae3c7397a0f95,
                    0x0daf664f185048fe,
                    0x5adddfdb4db07260,
                    0x2788785d9194d6ff,
                ])),
                Felt::new(BigInteger256([
                    0xd62585d5bc80d744,
                    0x49b001073201c18e,
                    0x3d6cc9046bfe12fb,
                    0x1f20b8a8e09e063d,
                ])),
                Felt::new(BigInteger256([
                    0xf5d5818d588dd8d5,
                    0xdbd71e10487f176e,
                    0xc52a752cd87b19f0,
                    0x1b042994b77babff,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfab9461374e9501e,
                    0x5ade8b4905245767,
                    0x8f54576eeae28a62,
                    0x2929468d5a74d741,
                ])),
                Felt::new(BigInteger256([
                    0x50de951123cb08c6,
                    0xe228d215baef8837,
                    0xd5454579b0c94d88,
                    0x19dda6135c6a4524,
                ])),
                Felt::new(BigInteger256([
                    0xb08908a2240991f1,
                    0x5b24e80fe50df8c3,
                    0xed25c5419e5ed8b5,
                    0x2e3a6e77238a0ccc,
                ])),
                Felt::new(BigInteger256([
                    0xf842f923bad7efdb,
                    0x01f22cff24086652,
                    0x173862de070c79c4,
                    0x1f8d0dbbf3a123b6,
                ])),
                Felt::new(BigInteger256([
                    0x3d8ff3e5b31d2b91,
                    0x21ac6acbd0e26ce7,
                    0x6eef37e5d1d1c399,
                    0x0d415a440858230b,
                ])),
                Felt::new(BigInteger256([
                    0x3288cdfcf7acd3fa,
                    0x665c46947a6e4e88,
                    0xf405dc3bc18f486d,
                    0x110ff30ccbbeed34,
                ])),
                Felt::new(BigInteger256([
                    0xb88b2639efc9a1b8,
                    0xdfe5273523fcf49b,
                    0x5a47c8157e9f2ecd,
                    0x27374d1276910777,
                ])),
                Felt::new(BigInteger256([
                    0xa70c70d47266bcf7,
                    0xcc508e2d94a6d14b,
                    0xcd14ceb875634989,
                    0x2a742e4701c4e76a,
                ])),
                Felt::new(BigInteger256([
                    0xa66da63b262f5f20,
                    0x9b6e552a1094558d,
                    0xb1b3b50012f2113e,
                    0x09d541b39872cd92,
                ])),
                Felt::new(BigInteger256([
                    0xe7884c1ff14680ec,
                    0xde5cc58222f82b67,
                    0x5a7b3b54d0d2d990,
                    0x25acf57588be882f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xefae781a1fd142ed,
                    0x9e326dda56d0f914,
                    0x0a091fd331bbb62c,
                    0x2ec6a2bd18c32604,
                ])),
                Felt::new(BigInteger256([
                    0x0fb22a1951bdaa6f,
                    0xd5ae5d0b88b78dde,
                    0x76a89e688e8a1cd8,
                    0x1b7b1b3b5fc9b4c0,
                ])),
                Felt::new(BigInteger256([
                    0xc8f1f7d60fdd1c64,
                    0x7c708e7cf33dcc13,
                    0x17c10adb69a746e6,
                    0x252ff69ad339bdee,
                ])),
                Felt::new(BigInteger256([
                    0x8b739066fd504bd9,
                    0xed94309fbac9861e,
                    0xb27393402737674a,
                    0x1f724a077fe64751,
                ])),
                Felt::new(BigInteger256([
                    0xc46eecd61ae535f0,
                    0x0f3eebb7d9e0941b,
                    0x68aa6a8aee749c95,
                    0x068ecc7176be8340,
                ])),
                Felt::new(BigInteger256([
                    0x741f38a6f509e3c3,
                    0xb133e306e2cdc57c,
                    0xba0a787da73f8081,
                    0x2c3c3a100fad883a,
                ])),
                Felt::new(BigInteger256([
                    0x86e2431732b28db4,
                    0xb477cf63fe1509a2,
                    0xbd59414491b4db10,
                    0x01edae6ac45ba83f,
                ])),
                Felt::new(BigInteger256([
                    0xab0fcf6d466f1f15,
                    0xf948ee879ee34612,
                    0x4d55f1ac6af860c4,
                    0x1a6fc7a091aa438d,
                ])),
                Felt::new(BigInteger256([
                    0xb7aeb4d232a912f1,
                    0x976bb28941caf37f,
                    0xabf29440e6f69c81,
                    0x0a97a77cd5a5c2a4,
                ])),
                Felt::new(BigInteger256([
                    0x77ca5b925055bb05,
                    0x25d2eb84caf7901e,
                    0xd03f097484f93e8b,
                    0x19991125c7989d6f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x252b3b8fe5b2c499,
                    0xa1385a7a54d923c0,
                    0xe5036611e31dcb6b,
                    0x16899fd37a7003da,
                ])),
                Felt::new(BigInteger256([
                    0xe789026ff64bf8f1,
                    0x9785dc9ee592db22,
                    0xb2e311a9c3e3aa26,
                    0x10b25b14e717cfe6,
                ])),
                Felt::new(BigInteger256([
                    0x277812dbf8c03688,
                    0x5eb981108d53e87e,
                    0x15c12a48274cc250,
                    0x261220c956fa3d79,
                ])),
                Felt::new(BigInteger256([
                    0x8e2d75c710cb7e2b,
                    0x99a8afe5dd750a69,
                    0x0d5ec89b79e773ef,
                    0x0c3958d591191658,
                ])),
                Felt::new(BigInteger256([
                    0x0e06cc8e23c5ea46,
                    0xe7b7eeab1eb868d2,
                    0x9a4e469847cc8275,
                    0x110846a84809fa4e,
                ])),
                Felt::new(BigInteger256([
                    0xf35cfa77d4b41393,
                    0x0142f072bd5ef2e3,
                    0x2faef648eca40d0f,
                    0x1e5dd2e7987fef59,
                ])),
                Felt::new(BigInteger256([
                    0xc23e7cf14b634562,
                    0xa15f5ae1b048a676,
                    0x2f0b85c994919153,
                    0x13823d3c668ebdcc,
                ])),
                Felt::new(BigInteger256([
                    0xd3825827b5b66675,
                    0x29fb62b553699b27,
                    0x7f258afb91a3bdbd,
                    0x0254e77ad5664aac,
                ])),
                Felt::new(BigInteger256([
                    0xfbae93a69532f39b,
                    0x5eb1bc63ed02e14e,
                    0x3a5c6825bd99ca49,
                    0x1d1f0ebae99c6ff3,
                ])),
                Felt::new(BigInteger256([
                    0xebad07e7e6746636,
                    0xace53126de8c4cd6,
                    0x8caebdcab5c7c284,
                    0x0413edeb6332c3a5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x98047c3749c359e2,
                    0x3a1ff620e8853c5d,
                    0x4c310823371c8462,
                    0x2d47d56e997d1d61,
                ])),
                Felt::new(BigInteger256([
                    0x81bade3335b084b2,
                    0x428f08c90e3a6925,
                    0x5f6628d379c5b9a8,
                    0x1df7b4343f608969,
                ])),
                Felt::new(BigInteger256([
                    0x787e33914e9be199,
                    0x6fcc9731e50d1c15,
                    0x968a2e7be25aec47,
                    0x1b230a076d37db84,
                ])),
                Felt::new(BigInteger256([
                    0xdbfcda9066934ef3,
                    0x85f79b8b918905eb,
                    0x1b52af01c1a82267,
                    0x22e0a8c6de6971d2,
                ])),
                Felt::new(BigInteger256([
                    0xc8ac89c68638ed31,
                    0x80f0f2052b704d27,
                    0xdd97fbad059a4eb8,
                    0x1e2353190fdbd49a,
                ])),
                Felt::new(BigInteger256([
                    0x01eb3716cf03df18,
                    0xbf310c631da76a5d,
                    0x601d5040895d3a5c,
                    0x068c8658bf2c6daf,
                ])),
                Felt::new(BigInteger256([
                    0x54de9f5f4f1c1887,
                    0x72c6696b081d8766,
                    0x69889e80a4bffef6,
                    0x15a57c8369ccc2ee,
                ])),
                Felt::new(BigInteger256([
                    0xee4b1002e5a98e64,
                    0x909c59d905d0b07f,
                    0x0c0783fe3626fa9b,
                    0x254f7961794f47a1,
                ])),
                Felt::new(BigInteger256([
                    0xbc89a2d8dbdda47f,
                    0x9ffe042c12664afb,
                    0xf3a41a92bd81cc3f,
                    0x290ee14cc94ff103,
                ])),
                Felt::new(BigInteger256([
                    0xdce566d13c5b4feb,
                    0x21ff20d01acd75c5,
                    0x3f85ed74d7a7aa54,
                    0x2458fb50a6bd965c,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x224240bfa6cece54,
                0xa6a21b5af80094c3,
                0x9e0cd5bd74437d35,
                0x300d2203cc240552,
            ]))],
            [Felt::new(BigInteger256([
                0xc13ffb88dfc0fd72,
                0x115ce95c0765d93f,
                0x7dad4704adfbd7b3,
                0x036a3b5fccfb686a,
            ]))],
            [Felt::new(BigInteger256([
                0x142c34b2e0c5c967,
                0x414089f82d4a69f2,
                0x878fb4b9c7d25fc5,
                0x1992537916554cea,
            ]))],
            [Felt::new(BigInteger256([
                0xcf722b18a0b0dda7,
                0x8ec470758506fadf,
                0x971059d3b652b792,
                0x30073c9c58c117c9,
            ]))],
            [Felt::new(BigInteger256([
                0x5b9c0758d1227219,
                0xe9f245d70174a53f,
                0x2fbaf4dd93237026,
                0x12a28698f863fcdb,
            ]))],
            [Felt::new(BigInteger256([
                0xbc3a013a2e1b1c9b,
                0xd2c9ee4b4b7ce5fd,
                0x3b761031c22fc1e2,
                0x2de8b7fe35db0249,
            ]))],
            [Felt::new(BigInteger256([
                0xa94b1315f5f64b57,
                0x195a77346f67c6b0,
                0xb447137ca698db33,
                0x0c50de10bfbe33c2,
            ]))],
            [Felt::new(BigInteger256([
                0x2774ed993e67b6d2,
                0x719063f7d6252380,
                0x345439c2923263a7,
                0x1fc5c78a4099f51d,
            ]))],
            [Felt::new(BigInteger256([
                0xf47e169475f17fe1,
                0x721939295f0f0722,
                0x4e7173cd23ac37c0,
                0x2d8c47d7430f9f5e,
            ]))],
            [Felt::new(BigInteger256([
                0x392799945bbdc7fc,
                0x04549bfd324581c5,
                0x1352ecacdb3be34a,
                0x0eb2c39b4f0c2821,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 10));
        }
    }
}
