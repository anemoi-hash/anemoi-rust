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
        // We can output as few as 1 element while
        // maintaining the targeted security level.
        assert!(k <= STATE_WIDTH);

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
    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
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
            vec![Felt::new(BigInteger384([
                0x2b79756e3cd21e7f,
                0x7b68c422ff14e0b5,
                0x9bc00750a636f158,
                0xcc2c09506fc714be,
                0x089c8aebfe4bae53,
                0x1035e3e284928dbb,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0xd76df25b9860db32,
                    0x7889584b7c765f2d,
                    0x7ce7ffca1a220f07,
                    0x8395585b7d568f88,
                    0xfa3a8c82a25cbd90,
                    0x04cd08e7b8243b99,
                ])),
                Felt::new(BigInteger384([
                    0x0c4e1299eac3e2d8,
                    0xcbcb7c23d2cce36d,
                    0xacf4d7a1463e704f,
                    0x1e37e2e8295d7d1d,
                    0xa349ea0372605f3b,
                    0x0866a8460dcc0480,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xdd2d7b78227020f1,
                    0xff70169f61189b2d,
                    0xbc24ede13777f66b,
                    0xcdd73e1e6af52c69,
                    0x91c756e512d0c4b1,
                    0x149c9808506b9058,
                ])),
                Felt::new(BigInteger384([
                    0xbb0e3cde0e6e5438,
                    0x40ea2d94a8d76cba,
                    0x072567ffcbff9bb5,
                    0x7ff96a74ea85796a,
                    0xa9973b3d686f61bd,
                    0x0e10499d3fe29bf0,
                ])),
                Felt::new(BigInteger384([
                    0xe380fb9e96fc0400,
                    0x9e8ababecc4ba488,
                    0x168ce89d787e05c5,
                    0xb9d66a0268a90a65,
                    0xa7970f5e87917b9c,
                    0x18a9142da7930377,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc31942b283d27f25,
                    0xb4284bdf2727fe5c,
                    0xcf7e075ef1155c92,
                    0x092bfe452241ceca,
                    0x903b0acb9d46ac74,
                    0x11fe38a6d0c4a179,
                ])),
                Felt::new(BigInteger384([
                    0xf3f7959623c3efbe,
                    0x2f6649370872c339,
                    0xb87ea5a8b993616f,
                    0xf37114c8b9919249,
                    0x65199241f71f19ae,
                    0x15643becf4d71dfc,
                ])),
                Felt::new(BigInteger384([
                    0x57a619d898f53a36,
                    0xb7c0cb746d45df98,
                    0xa56e83679b50cd1a,
                    0x08e09f75fc07383e,
                    0xa92e54c2633bc880,
                    0x081118fffedc0e8d,
                ])),
                Felt::new(BigInteger384([
                    0xecb05b96c6a85ba9,
                    0x297206fa6f6dbcc9,
                    0xd2ad3f002030b23b,
                    0x736cfe6c1015df50,
                    0xc13a2c449e572ff9,
                    0x13c9280ee242d405,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa41ed8031e273043,
                    0x1bb24fd36add2832,
                    0xb3f54e748163688c,
                    0x9da5a8780bc6fa83,
                    0xbcfe462079bcc6ee,
                    0x06541239d7fa9105,
                ])),
                Felt::new(BigInteger384([
                    0xcdf36ecd13eac88d,
                    0x2d55b6519d4a0804,
                    0xe51cd0bd00b5abdb,
                    0x73a78722e76c5738,
                    0x0e91c096dfea2b82,
                    0x1331b5ee7dce0b63,
                ])),
                Felt::new(BigInteger384([
                    0xcf8b7df71a10a86b,
                    0xae31ccccf5dca8da,
                    0x578f9f1ee27178d8,
                    0xb9bc9cf9ca5b4a88,
                    0x7bf9bb0c0453b592,
                    0x184f831fff323818,
                ])),
                Felt::new(BigInteger384([
                    0x2889bea8bc519de1,
                    0xdd2338dc77b07b46,
                    0x07530457f93792a4,
                    0x75065d6a8614bf8b,
                    0x44f8de6c71fe2f89,
                    0x010a862e116a7d6b,
                ])),
                Felt::new(BigInteger384([
                    0xd1f9760205ba6358,
                    0xf2a81c49bcf341c9,
                    0x1c19d28ebf06e105,
                    0x1bebccc32f01f43c,
                    0xc0401c5580b2659f,
                    0x113fe416697c726d,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa137382c149b75e3,
                    0x01deb0f27cede6f1,
                    0x663052201a98fdb4,
                    0x9543bd88bd3661fb,
                    0xd2eb829f7ca744cf,
                    0x10a3207bf4e0c0ad,
                ])),
                Felt::new(BigInteger384([
                    0x1cea5152b1f449d5,
                    0xecca5595ebe92415,
                    0xd9177f42429bdc77,
                    0x5db0b019f59dd96f,
                    0xe1644966f0d01a1a,
                    0x0279f92d0adf346d,
                ])),
                Felt::new(BigInteger384([
                    0xe52a6f9c9ded8c3e,
                    0x4c3b7cfaa7866248,
                    0xcaa7605217afc5aa,
                    0x5b5752471ca111ba,
                    0x1e4ce9eac02b0318,
                    0x0770804c0d5aa638,
                ])),
                Felt::new(BigInteger384([
                    0x7b10f9de23c7fb90,
                    0xe5f2350190617c3e,
                    0x289b4c7324bc91f9,
                    0xa94b1c86209be8e5,
                    0x1a7a128c6436d00c,
                    0x190e70a30cbf2025,
                ])),
                Felt::new(BigInteger384([
                    0x619154af64fc2ff3,
                    0x1518b7f01ad084dd,
                    0xa7ee4725d09a6f3b,
                    0x4e70ac7aca0c5fc3,
                    0x3cbd5548aad4bf85,
                    0x000b03911fa48908,
                ])),
                Felt::new(BigInteger384([
                    0x8901495291630cfb,
                    0xa92363c947dcb6b1,
                    0xf4b2eed704376386,
                    0x5c9e9fc079e68e0f,
                    0x06a3f65863c5c48f,
                    0x07b2e9de6cdc439d,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x146c3244ecc5dac8,
                0xce65f3b7203a6d5f,
                0x230a32369c676906,
                0x075096e6d8931333,
                0xc45c6846979caa96,
                0x0dd15d21ffce3dc2,
            ]))],
            [Felt::new(BigInteger384([
                0x94febde6093ac7a9,
                0x205d96a375cc35f3,
                0x813780c868fd7d76,
                0xc6efbd8f0976283d,
                0x678f07337a9aee6a,
                0x090cebb5d7a227e4,
            ]))],
            [Felt::new(BigInteger384([
                0xe17089c0d5bd1d7d,
                0x3bf65cd976b256b6,
                0x8260cd233b94e84b,
                0x5a6dfa7928a42379,
                0xb8622876fa913b2f,
                0x03bd18cee56a2686,
            ]))],
            [Felt::new(BigInteger384([
                0xd6d8518d27e3a6c8,
                0x48a23c37225ecbb6,
                0x05c61eecdc60e55b,
                0x3e1ca3c677dc09e4,
                0x3908777e798522ce,
                0x0e4b1f51afec0e25,
            ]))],
            [Felt::new(BigInteger384([
                0x89629bfa744f73f2,
                0x5cbd055d596abd10,
                0xb741d4e0af6090fd,
                0x4b93ce873bf07336,
                0x709f20bf5d144a21,
                0x115fe6064f5f75b8,
            ]))],
            [Felt::new(BigInteger384([
                0x5561523dd24d1fa8,
                0xb85c335655179b98,
                0x633ee023e98e349d,
                0x83406c2f4245d6c5,
                0xc56917043b8177d0,
                0x10d459b28f3f7eac,
            ]))],
            [Felt::new(BigInteger384([
                0x7ebe25b61950b3f6,
                0x1066db48beedf326,
                0xbca484f6961a5ed5,
                0x3a18a6fee2badde3,
                0xedeff7d761b5dfb7,
                0x0d984b64bfc90851,
            ]))],
            [Felt::new(BigInteger384([
                0x7129203983b303db,
                0xea8b470c421784a9,
                0xe5ddea291d4e4c35,
                0x0e0e59f29ea7b544,
                0xffbc5eb7c6eb23be,
                0x04a0ee9ef6888115,
            ]))],
            [Felt::new(BigInteger384([
                0x772492d5d55b7fd5,
                0x2ff2efd82596576f,
                0x8e7eb0bf0e9a7994,
                0x47c81b2969044eb9,
                0x92b809be880034e6,
                0x01533de540ff22eb,
            ]))],
            [Felt::new(BigInteger384([
                0xd0993156126bb454,
                0xb13f13a910c49b51,
                0xeb1a6015eb0eb4d6,
                0x79696f1a3578b134,
                0xf8e08535772be81f,
                0x0c58a3fc1bddbb97,
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
                Felt::new(BigInteger384([
                    0x90cb5ed557b3f1fc,
                    0x50724fdb9d28faaa,
                    0x9ef9574db4587a7a,
                    0x997844183ade512c,
                    0x721f13921b5210a9,
                    0x0395b348526f8d17,
                ])),
                Felt::new(BigInteger384([
                    0x0413be83a74ba436,
                    0xd65115ce7f21ae00,
                    0x3909871bee25dc61,
                    0x3d275fb9f0d97de8,
                    0x7189a639cae018d4,
                    0x0c67665ba4f78782,
                ])),
                Felt::new(BigInteger384([
                    0x35009753b5902c48,
                    0x6b1aad42b3b45674,
                    0xeabbc713eaabc4ad,
                    0x62c90004860eae5e,
                    0xc6743f53b9fbec2e,
                    0x0626f40c18780e2a,
                ])),
                Felt::new(BigInteger384([
                    0xa97fdf474d587913,
                    0x89de1aa2b3b11976,
                    0x94faed2a829c6a6d,
                    0x2a10aac416b93300,
                    0xd7f53fad1ce26151,
                    0x1874ef58883f2904,
                ])),
                Felt::new(BigInteger384([
                    0x832272b78ee4590b,
                    0x7e6066c28db1b472,
                    0x8b28359958ed20d1,
                    0x44aada92efd6b8a7,
                    0xad49691777460205,
                    0x1373f679999352d7,
                ])),
                Felt::new(BigInteger384([
                    0xa792c5f2c843691c,
                    0x1adb32422b48c2b6,
                    0xaf9a780931b08298,
                    0x1b6e7faef29cdc83,
                    0x98962d5c8c569925,
                    0x06ea2cd8eb5af900,
                ])),
                Felt::new(BigInteger384([
                    0xd8b2c0926d243ea7,
                    0x55a841effa6a9698,
                    0xc0ed2b5f4aadc7f2,
                    0xff3e6b75b6b618cc,
                    0x021a0bea61eb08fb,
                    0x07ce2be906b76f95,
                ])),
                Felt::new(BigInteger384([
                    0xfa10a79b74e7f18e,
                    0xdff3f4846070b3bc,
                    0x645bf31e314d37f2,
                    0x14d90032a9fcb5ad,
                    0xfeba715b16726e97,
                    0x0037d92a44e9a519,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x773e7f2a8b4e085d,
                    0x15d4dbe61b431d0f,
                    0x266bc1dc34732794,
                    0x7070e7027a281303,
                    0x37848277d295cc82,
                    0x10e30505d18f835d,
                ])),
                Felt::new(BigInteger384([
                    0x88089bc17fc904c6,
                    0x02be84c07dbd847e,
                    0xf669c44d1a9e2406,
                    0xe35a5f8903431d0b,
                    0xbe70de4de53245f8,
                    0x03c9761d6fc0780b,
                ])),
                Felt::new(BigInteger384([
                    0x3c758302bc7eb3db,
                    0x3949c6cca6617fb5,
                    0xfd6b6d78a9f9c2c9,
                    0x48da343da573cd3c,
                    0x872162abd5123966,
                    0x09c2e8a729f5616c,
                ])),
                Felt::new(BigInteger384([
                    0xcc6eb40773aba698,
                    0x8a9198f63ed4ef0c,
                    0xb2b78bd9c8e07351,
                    0x6a0d43a3cce5ae13,
                    0xa67055c5bf2fb0aa,
                    0x189ef7020048c27a,
                ])),
                Felt::new(BigInteger384([
                    0xd5d58fd534292db6,
                    0x10b95c58fcb8a749,
                    0xbd8b19de88167d82,
                    0x3b801dbded554c81,
                    0x2b1c43b0644c130a,
                    0x1a00bd09dd0fd590,
                ])),
                Felt::new(BigInteger384([
                    0x243af8b8f12e0fef,
                    0xd78287a75f9371f6,
                    0x9ba2d4546819c721,
                    0x9c4de7967211a64f,
                    0x0469ead1b0350f0d,
                    0x05a5d3128f603ac6,
                ])),
                Felt::new(BigInteger384([
                    0x1b3af1cca3e87a2c,
                    0x9c5a275ed05429be,
                    0x6643b21e6daf6c35,
                    0xfc7c4883d709c8a2,
                    0x831f743ee76e896c,
                    0x0a85dae5d2e75601,
                ])),
                Felt::new(BigInteger384([
                    0xcf3af5a4be602102,
                    0x7be302e2a39512ff,
                    0xd286061f91170e7a,
                    0x8c4a0c70561532f6,
                    0xd92fbaf2a52fe046,
                    0x17f4b674735b9f40,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x015b9ff4fb9347e5,
                    0x04ee3070c3cf1922,
                    0x999c786c2865a918,
                    0x2ced323f4d71058e,
                    0xdf9aaf8051aab161,
                    0x147da1d7b416b6d3,
                ])),
                Felt::new(BigInteger384([
                    0x73caabc1c3e4feb9,
                    0x63a902278718b0ad,
                    0x7e6d2ab16b69d84b,
                    0x6a38cf30c1be35f5,
                    0xc6e58b648cc99032,
                    0x16a99a32f7f841fe,
                ])),
                Felt::new(BigInteger384([
                    0x3b9dd308767188b1,
                    0x3a759005c04fc4b0,
                    0x6a960838100784a1,
                    0xb5912725431fe05f,
                    0xbc54aff2054d284c,
                    0x18867956d8b1eecb,
                ])),
                Felt::new(BigInteger384([
                    0x05fb682b161b181a,
                    0xcc7c736eb8e75348,
                    0x95dbe793c9659421,
                    0x11e87b788b5eaf69,
                    0x535ecfdd3b2cd89b,
                    0x03693b0d282a2fb8,
                ])),
                Felt::new(BigInteger384([
                    0x4cc92e1a43a9e53d,
                    0x655a62ace8c84eb1,
                    0xce6dd94461c4cb45,
                    0xd31716d12b7e724b,
                    0xb1b975e60f016e47,
                    0x01d83bbdbbac6ed9,
                ])),
                Felt::new(BigInteger384([
                    0xb7eaae5f0d56a3b2,
                    0x851620a857cc9d00,
                    0x4d599ea6ccd580db,
                    0x518c4e6b1a3b8b8b,
                    0xcf8b029248a1fd51,
                    0x01e05dc28bc0316b,
                ])),
                Felt::new(BigInteger384([
                    0x9e09feb764f6e239,
                    0x515d8d1dde701f09,
                    0x583229bc13367be6,
                    0x8f146daa568cdeed,
                    0xe5d84f738ff8a044,
                    0x0aeb457d4bb41d29,
                ])),
                Felt::new(BigInteger384([
                    0xcc25277b6e0aee28,
                    0xd8a73c7f9566f7c0,
                    0xe579b2ce701cb5bb,
                    0x23d67c847a588454,
                    0x69c4bf2558c1ac02,
                    0x11afe9dc2f136c82,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf4f254510497eaf1,
                    0xfbf1912588825440,
                    0x35b55c493e175d8f,
                    0x7f5fae2b2207456e,
                    0xce3fe474b87bfb49,
                    0x14378503a2fa10cb,
                ])),
                Felt::new(BigInteger384([
                    0xa29d778ce76720d3,
                    0x64f4ce56c0d28abf,
                    0xe35b7336a138ba43,
                    0x0e99489973711171,
                    0x2c00210aa966dba6,
                    0x12c7a26a61d3aeee,
                ])),
                Felt::new(BigInteger384([
                    0x920753cf0ffb9af2,
                    0xed1c9cf5ce84eb53,
                    0x87f79a747c6fd6df,
                    0x743da9888bb4d7d1,
                    0x6a0f2c0c964f99ef,
                    0x03f2046a7360269d,
                ])),
                Felt::new(BigInteger384([
                    0x8daf6ba1cafde29d,
                    0x47855f5abc3e070c,
                    0x5476386f131f2966,
                    0x4c56ce78e8cde0e5,
                    0x8f6246635580f047,
                    0x0345c2e953c71a43,
                ])),
                Felt::new(BigInteger384([
                    0x58c288681ed86c79,
                    0xefad3cd90c508bcc,
                    0x5d7f1ca476c2b1df,
                    0xa477830fd11a95e3,
                    0xe959eb043fcbfe9a,
                    0x1569e02847673a98,
                ])),
                Felt::new(BigInteger384([
                    0x472687f983527ec1,
                    0x65964c64e46232ec,
                    0xf51c0dcde30ef576,
                    0x2d92aeab525cae05,
                    0x1a2529c0aa0d9eda,
                    0x0452f2cbeeda7025,
                ])),
                Felt::new(BigInteger384([
                    0xc592196d5f78e2e8,
                    0x1e3305bdb798dc0d,
                    0x037a0a46922577d9,
                    0x79f616eb94bc78b8,
                    0xa124240ec4c789aa,
                    0x12647cfb5b8e420b,
                ])),
                Felt::new(BigInteger384([
                    0x79da83edfbe22b09,
                    0x03a73466fbbd00cb,
                    0x83a155db7fdf27b7,
                    0x53bbb03b74ae1ab0,
                    0xe5fe59ae095ee322,
                    0x1610b9b61d6079eb,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x0376d33bb9c8e3b9,
                    0x678972331b8be149,
                    0x16946437bdca5241,
                    0x1ffd9fe217f650bc,
                    0x83cdf43cb0bff8bd,
                    0x0784dd515cab3d61,
                ])),
                Felt::new(BigInteger384([
                    0x89704b2a37d8dd93,
                    0x369e23999a8d0cdc,
                    0x6ecf6a85b07c97a0,
                    0x081d08f3f9c5489d,
                    0x864fc06c222aa878,
                    0x08601aecbe44b718,
                ])),
                Felt::new(BigInteger384([
                    0x015f808e0dbfa3ac,
                    0x488e4801a671f920,
                    0x95b507e0325a9169,
                    0xd5490b98b0cd94a8,
                    0xb5dac9c0ddef8474,
                    0x07c957d5aea5f2e1,
                ])),
                Felt::new(BigInteger384([
                    0x18b02fd6555520ca,
                    0x945d38a6b4795228,
                    0xe19d6c0fd8650205,
                    0x7b24520f299aec29,
                    0x644e76d6b02c571b,
                    0x1648fa7dc8f0f1f1,
                ])),
                Felt::new(BigInteger384([
                    0x8eb6dbe8b7a553a0,
                    0xacfa8bfbadaf323a,
                    0x38344d93e50dec67,
                    0x157f69a2e2e6166c,
                    0xc923bd0184ef534b,
                    0x19accae1bf926ddf,
                ])),
                Felt::new(BigInteger384([
                    0x5e141738420bb22c,
                    0xe4b99dca71a9f418,
                    0xaf6b46db558a8e5f,
                    0x9c0f74bba1162c4e,
                    0x70d74cc5f22d7df4,
                    0x0a982ade1d525ebd,
                ])),
                Felt::new(BigInteger384([
                    0xeaedc50a1afc1f45,
                    0x9e04e04fae0aea83,
                    0xc617a8549eed2ad7,
                    0xda3f4ac0d57a8dfc,
                    0xafc72908202fed93,
                    0x117f5d92467c4deb,
                ])),
                Felt::new(BigInteger384([
                    0xa51ca3e5a7aaef37,
                    0x2877a464910d81c8,
                    0x62e0b8e3e5f2cfd8,
                    0x84af0802af1fb777,
                    0x44955cd2f7230cc0,
                    0x14728118629750de,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xcb39f288b6b44981,
                    0x327462d1c1e702a9,
                    0x747140809db204a9,
                    0x8f88a6d7064ef3f2,
                    0xe677f82991d95288,
                    0x19096e56b24397e2,
                ])),
                Felt::new(BigInteger384([
                    0x353745d9cc58466b,
                    0x7646e5ee5a86e916,
                    0xa6860f6098f86bc9,
                    0xd462fc66509a166b,
                    0xd9ea30403507bc64,
                    0x0c3c666cedd86a21,
                ])),
                Felt::new(BigInteger384([
                    0x5a0caaf1e7bb0554,
                    0xb0e5ab0fffc42675,
                    0xae498d082d05cf84,
                    0x949ce1bc87022ba3,
                    0xf4ff6620e7e2d9ac,
                    0x0f027ed8ba1c9679,
                ])),
                Felt::new(BigInteger384([
                    0xb23b66fc76c5bbf1,
                    0x594a36774b330d11,
                    0x60a6c3eba7e81b13,
                    0xf298a936ce867a21,
                    0x52322bb39345df1d,
                    0x00ef924d3c92e753,
                ])),
                Felt::new(BigInteger384([
                    0x4f0d2e2167bc782a,
                    0x689130e8bf7bc536,
                    0x26b45883b1276809,
                    0x3c08c6b3ac1b93a5,
                    0xaee994df4cf2c397,
                    0x17e8cb995f2ed1de,
                ])),
                Felt::new(BigInteger384([
                    0x87508e2f9e240fd5,
                    0x87d9280ef04ed444,
                    0x1943db2546e3a464,
                    0x98ed3af906a4042a,
                    0x15b7cee2bb8c8433,
                    0x17218abacc588bdf,
                ])),
                Felt::new(BigInteger384([
                    0x539cde5737e8aba9,
                    0x14532e8670463842,
                    0x0612fe86e6d4c0b4,
                    0x78f7e3f197e44d18,
                    0xed0ad4a1084b5254,
                    0x0fefe15e137b0d19,
                ])),
                Felt::new(BigInteger384([
                    0x34a56009c0b4f94d,
                    0x56771603c7b58640,
                    0x7c0992f8aecf082b,
                    0x3c341e37c0ece09b,
                    0x365877cc72dfe2c2,
                    0x155c7ebca34094cd,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0x81a2a2126a6a1f1a,
                    0x5a5a18be3381b860,
                    0x857f68f483434c28,
                    0x83b4f7a903ff5895,
                    0x1a1ced8e2ba2f30b,
                    0x163f7109c1ad7523,
                ])),
                Felt::new(BigInteger384([
                    0x236be4de7add6a8f,
                    0x9f56f0047b55702a,
                    0xe6323800f9321e0d,
                    0xe0e694ebd9a37508,
                    0xa1e8e541677a568c,
                    0x15de47a140abc671,
                ])),
                Felt::new(BigInteger384([
                    0x06718bbb9c2b9183,
                    0x8480b57bf8bdd7e2,
                    0x0f1be9af5a55042c,
                    0xe2b26989cb3cf84d,
                    0xf687f900c2245c38,
                    0x0edc19ced8bdb85f,
                ])),
                Felt::new(BigInteger384([
                    0xe2b06d7a402ce0d0,
                    0x30ccf1d3f02fb078,
                    0x4120735653142471,
                    0x976a1296db4a1bf6,
                    0xf2a7dfded297207b,
                    0x0313f31d7a7e9f41,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x157fbf9522e2bf7b,
                    0xe55e8970a4808367,
                    0x6f7d9ad9506b11a6,
                    0x1602d276ac318657,
                    0x9767f53b12a75b65,
                    0x1750aedc82b13c0b,
                ])),
                Felt::new(BigInteger384([
                    0x19c848be178e4eb1,
                    0x13aa7ac0e7189a88,
                    0xb9dbdf6be3b8315f,
                    0xe0a18c09f1115b13,
                    0xfe8e6915882b6ad1,
                    0x04f826ca40c93527,
                ])),
                Felt::new(BigInteger384([
                    0x4b3fc30ac42baded,
                    0x0bfdd899995f888a,
                    0xeb33e3fd3da853b2,
                    0x950f94c015e1cd13,
                    0xe408651e134b6223,
                    0x08a58e2f227d9084,
                ])),
                Felt::new(BigInteger384([
                    0x8d6e26321df3d9f8,
                    0xf55ee13fc1d55963,
                    0xe48fe15f7d0e4634,
                    0x54232024af33b5d7,
                    0x431625679df74e53,
                    0x0ebe7ac8a661e42c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4036d6f3ad70bc47,
                    0xb64ac7ef76514dfa,
                    0xf1279cfb97acbc51,
                    0x56c399abb72dd386,
                    0x76c5e88efd948770,
                    0x013b79810e0a19c5,
                ])),
                Felt::new(BigInteger384([
                    0xbb601f6f6f2bc879,
                    0x689fa28b6bf6a83c,
                    0xa5fb9caa40132bda,
                    0x4b4173921f42a6fe,
                    0x2b2c33d05d333edf,
                    0x0f0fee85892131ae,
                ])),
                Felt::new(BigInteger384([
                    0x4ca47129ecacec56,
                    0x8ad43ddf9d0ec91e,
                    0xd6aadbdbc2fa36c5,
                    0xcb7ae1fedd88b45a,
                    0x18e621361db295b3,
                    0x1051a3c69f70c86a,
                ])),
                Felt::new(BigInteger384([
                    0x9f164317958ee900,
                    0xcb11387b1854be5d,
                    0xb0461cbcdcd5eefd,
                    0xcf20cbb8c7a52d47,
                    0x87dcb722c83f33fa,
                    0x13e7d5cd6805a5f3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe95a807efddb6070,
                    0xe0b5ea56c25e1560,
                    0x510049e8b1348b6f,
                    0x78c65ea776876fc3,
                    0x69b95c2de502848a,
                    0x0aae0325b34a446a,
                ])),
                Felt::new(BigInteger384([
                    0x07b85257cd6f4d41,
                    0x4e0bac86211037fc,
                    0x7df13dd315ce2cc8,
                    0x7d4781467ceadac6,
                    0xc46910df960ce10f,
                    0x0e6fddd130035b5c,
                ])),
                Felt::new(BigInteger384([
                    0x730ab6dd8f4b5f82,
                    0xde585403e8194017,
                    0xfe99e5d9ac885726,
                    0x2086db15d7205e6f,
                    0x47cebc012191948d,
                    0x13d68871a2d2fc72,
                ])),
                Felt::new(BigInteger384([
                    0xab2c204774d4e418,
                    0x4ab814ae393f47fc,
                    0x04fb38b1ffff2443,
                    0x51dff687f6ed3965,
                    0x0888378076bc831f,
                    0x1015fe95b1f06241,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x85df38c380bee1c7,
                    0x0f673e9de564132e,
                    0xa5d8bc7f638ac0c2,
                    0x6c39603c7df6730e,
                    0xe2374d85e3b27fe2,
                    0x0f6a12126628db6a,
                ])),
                Felt::new(BigInteger384([
                    0x8c2f5d5a481dfda7,
                    0xa8787758964f621e,
                    0x7503708b29fd5f9c,
                    0xa99b927ae25b3056,
                    0x5cd43fe331d2b49e,
                    0x14ac9638118b3b32,
                ])),
                Felt::new(BigInteger384([
                    0x9ff22f2ce6e47139,
                    0x24410441f7665cf1,
                    0x73ef30ff36bbf61c,
                    0x5d2d1dc9a5cefd49,
                    0xcf2e537674d743f5,
                    0x0dcfc7ddcb22f104,
                ])),
                Felt::new(BigInteger384([
                    0x2119a6b9b914badd,
                    0x2ba1bd0f832e510d,
                    0x33d04a75aef05d65,
                    0x9dbaee21573dd8cc,
                    0x88fc6554e9cee284,
                    0x0da6229ac46b95d4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xca6c3fe9de0d1ff7,
                    0xc9d7e60e61e63909,
                    0x305e6ceb0637d528,
                    0xd4815b28d601cc46,
                    0x3a676e75a07eb69a,
                    0x09da66030d6d1b60,
                ])),
                Felt::new(BigInteger384([
                    0x1fa78cb0f1522d7d,
                    0xa5286cec2ea4871e,
                    0x31b662263e5f0ace,
                    0xd29f97283b9bf36c,
                    0x23da7b7550614c6e,
                    0x08b9dd749433f271,
                ])),
                Felt::new(BigInteger384([
                    0x56a534a9b23149ad,
                    0xd59cbce8faac4c21,
                    0xb5e7a3ef7554ad07,
                    0x104650293900d38e,
                    0x601b411bd833c917,
                    0x166388c388af4317,
                ])),
                Felt::new(BigInteger384([
                    0x1f91729f8e72827b,
                    0x784499ff18f0f04d,
                    0x80ba8b2afc0f6fbd,
                    0x21393d6437a2fe2e,
                    0x85573b6d4fa2f101,
                    0x13e8869c39c988ef,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7ff23f3ee0263761,
                    0xea194d1f511de8de,
                    0x485a12ca8061cbf0,
                    0x9a2ce205c6d2ab8a,
                    0x4107767663364513,
                    0x18becf8b015d2ad4,
                ])),
                Felt::new(BigInteger384([
                    0x7119700ae6113409,
                    0x286992d62c10007a,
                    0x25de34c6c6a09c59,
                    0x9aeea9f09fc90a0b,
                    0xb43b94aa2d7f88d3,
                    0x063a7014eaf25437,
                ])),
                Felt::new(BigInteger384([
                    0xd5026f0922fc5091,
                    0x398665177c6a6da1,
                    0xb80a59abbff84123,
                    0x20936ca1fbf3b291,
                    0x7fcb4e0cb640b76d,
                    0x068c32bdb6b0e583,
                ])),
                Felt::new(BigInteger384([
                    0x0c848c40f9cfdcaf,
                    0x7e37bbfd5b3791ec,
                    0xb3c8f8640ce71165,
                    0x39c162453e74656e,
                    0xf15ba6cac31f8e5b,
                    0x0e0a29cee18d1d23,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x17ff1821244ab577,
                    0xc106d514b6fc53fa,
                    0xb4d3039668d9c8c9,
                    0x709edcff621827a7,
                    0x97dea9a0a8fe13c3,
                    0x0176371b0dbefdc8,
                ])),
                Felt::new(BigInteger384([
                    0x55c690275ecce2b8,
                    0x3ced87ff1b2e5363,
                    0x0536972eba967768,
                    0x625b5b3883343357,
                    0x54380e2a96b84eac,
                    0x099f69010ca08550,
                ])),
                Felt::new(BigInteger384([
                    0x4bccfe43a8baf8f6,
                    0x49438c718bf762fb,
                    0xdffc5f3be33af8e3,
                    0xc43699158a2db2ba,
                    0x563fd3c7d1a3e392,
                    0x10a3153a12cfdabb,
                ])),
                Felt::new(BigInteger384([
                    0x3f88158638dc4112,
                    0x3f3b1c930931ab55,
                    0x07c67a3bfdb806c2,
                    0x3ff701323ced2ef8,
                    0xa15e72d3b9f56027,
                    0x09dcc3217ed4be55,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x5f4917f92779e40d,
                    0x1dac416b4d519b1c,
                    0x42da165c0e8528a8,
                    0x4732091cda01ec05,
                    0xc3f1524cdf7343a5,
                    0x00c1bca13074af18,
                ])),
                Felt::new(BigInteger384([
                    0x241972ca32c4c8fd,
                    0x1098f6ce357d61f8,
                    0x73e7aac782d36362,
                    0x9ee0ba422a6b4d10,
                    0x5ef691dc6cd607bd,
                    0x11bd5a82e1d63bc1,
                ])),
                Felt::new(BigInteger384([
                    0x46fb6e61a9eaa436,
                    0x679000b0353c9c4e,
                    0x9ddf21c6c4e49178,
                    0xf017c9f14deb27ca,
                    0xbbeadb97c969b155,
                    0x01c9819e8a93e612,
                ])),
                Felt::new(BigInteger384([
                    0x108b6712e7c2cab2,
                    0xadee21cb61c0039a,
                    0x570a0350aee658b2,
                    0x800b19d83bd32ed1,
                    0xe730603293eeff7c,
                    0x0a37a41af0273149,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x42c55510205f698a,
                    0x12c621366a235a3f,
                    0xe9d73f3633e57ace,
                    0xdc1d7075ccd5c75b,
                    0xe14ec3ab5971af9e,
                    0x047a2dfcb209039b,
                ])),
                Felt::new(BigInteger384([
                    0xeb141f02881cbfc7,
                    0xc3eb8f11a810ddb7,
                    0x09de425095880716,
                    0xee837f83343d2d63,
                    0xd70b2217758c0f6e,
                    0x107a1d7575511d11,
                ])),
                Felt::new(BigInteger384([
                    0xc7bc74e946d587e1,
                    0x6740f64e3f8c1ed6,
                    0x21071cfdc87a66da,
                    0xff3d26ea829be507,
                    0x7fea5bd3d1224ff9,
                    0x06f657de275710a1,
                ])),
                Felt::new(BigInteger384([
                    0x40baea5d5111d37a,
                    0x827b270f580b36b8,
                    0xb92f949d000a954f,
                    0x504b14f655974f70,
                    0x1c6ed189c60f04ea,
                    0x14c71e815e860458,
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
                Felt::new(BigInteger384([
                    0x0e6f34963e4b07c3,
                    0xa0c350f1a5597704,
                    0x6eed2fe8b2a8ba5a,
                    0x423d20c3ce0071e3,
                    0x3a4f0b8fc4830546,
                    0x0034abc078491851,
                ])),
                Felt::new(BigInteger384([
                    0x5e07ab693f3651ea,
                    0x22d489535176cec2,
                    0x2741cc152c304a5d,
                    0xa216b09821f8d4d3,
                    0xc90ce4a67bd66cab,
                    0x03f380aa5da9ff58,
                ])),
                Felt::new(BigInteger384([
                    0x74771e0622091785,
                    0xe2f3ddc38246cb74,
                    0x07a203d28c98a592,
                    0x47665d957d80b506,
                    0xab66bd34a458b59e,
                    0x11815f81805b695f,
                ])),
                Felt::new(BigInteger384([
                    0x8608787aa24e3efb,
                    0x2f2c577d9a44eb8e,
                    0x03bf6b65d8afc4d0,
                    0xe0aa0fbfbf8498d6,
                    0x9f06c14ec9498ceb,
                    0x12741485754d8db3,
                ])),
                Felt::new(BigInteger384([
                    0x250632f25aca1927,
                    0x5e9e4e51f106589e,
                    0xe90acaf218cd1328,
                    0x22493b44aebda245,
                    0x2403e58a6de6807a,
                    0x06daf550d6a3ed94,
                ])),
                Felt::new(BigInteger384([
                    0xf201202afc52d9d2,
                    0xf3477771f510e454,
                    0xc0b121a509a49d91,
                    0x730ff82beecb1faa,
                    0x20800021d993f5ae,
                    0x0a527e1b40b8d452,
                ])),
                Felt::new(BigInteger384([
                    0xb13e3a17016bd241,
                    0x5cab38f185559176,
                    0x17a115a204f59502,
                    0xd2ae0ade8b4f5057,
                    0xf9fccbb722995536,
                    0x0b1a305ca3e5d550,
                ])),
                Felt::new(BigInteger384([
                    0xcc5ebedf9a9483c8,
                    0x04aea7b724be130c,
                    0xaf1dd82dba969ad9,
                    0x84539aedf445a3cb,
                    0xb8e9a402c53c7f26,
                    0x0132224d1bb0d9d0,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x18b35b648580a423,
                    0x841851fc4126e0f4,
                    0x1ef43d6a5085b2ae,
                    0xe595f250da889cfa,
                    0xcad55e8286c9939c,
                    0x0c11371ffc54b1b1,
                ])),
                Felt::new(BigInteger384([
                    0x6690645b6c388244,
                    0xbbfca00078408ea0,
                    0x247b6e3a194d0a48,
                    0x2a6c5fedfc6f14bd,
                    0xf7972272199a2dd4,
                    0x1188148476e76bed,
                ])),
                Felt::new(BigInteger384([
                    0x1e790bb97883e6fb,
                    0x7a2725be5d9408de,
                    0x1e20af42f4d4907d,
                    0xa6a1aaea097da4f2,
                    0xcc1eb3b203fe4d63,
                    0x016b89990d4e795b,
                ])),
                Felt::new(BigInteger384([
                    0xd578fbc99320658b,
                    0x92ed2e22e13f87ac,
                    0xd844882ef95db66f,
                    0x30e3bceef22f3a9b,
                    0x5402df7318db70f9,
                    0x0b49879c0530fc2d,
                ])),
                Felt::new(BigInteger384([
                    0x6bcd506bebd2fdbe,
                    0xf529fcf1658246dc,
                    0x7ac13d19d2444195,
                    0x15eb95b3021d4664,
                    0x6f9ac81150225237,
                    0x1501aef42a990118,
                ])),
                Felt::new(BigInteger384([
                    0xfc3073c544aa2ce4,
                    0xfa2d2b5d26d4122a,
                    0x22f0ac3b9e60a624,
                    0xaed409b002fbc851,
                    0x715e837b510f6175,
                    0x100b53a4370ba7c8,
                ])),
                Felt::new(BigInteger384([
                    0xbf721e6f104c19fa,
                    0xb71ed19bc529ea77,
                    0x35b902a1188c2d8c,
                    0x9f03a85c75144ab0,
                    0xa9599f7424e16e12,
                    0x055eac738c0a300e,
                ])),
                Felt::new(BigInteger384([
                    0x264cdc3d49f214ef,
                    0x90d0ba826a87ae0b,
                    0xb190eebd78be6f9b,
                    0x0f52940e2945de9a,
                    0x2eac277d33f2a97c,
                    0x14599b358f0db4e3,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x9dd940b6f835547d,
                    0x8d44549f6a0e341b,
                    0x7cf5c0fa14e52279,
                    0x4c183dd2409d0cbd,
                    0x53ed1f16a0d5ee87,
                    0x0e0599014ec5133d,
                ])),
                Felt::new(BigInteger384([
                    0x68d6c3c9d20151b0,
                    0xe2a4471beb1f6a3b,
                    0x0a4b17c5ee1a468c,
                    0xb69fb0d5acdf8367,
                    0x925c09bdd4a5e1f2,
                    0x15b9825b86b24844,
                ])),
                Felt::new(BigInteger384([
                    0x1e667f0edce42377,
                    0x49bcdefa1d9c34c0,
                    0xc3d57a5606385ef2,
                    0xa1f2d54c416203aa,
                    0x71ccd7c706a379eb,
                    0x035a8fbe633c0c40,
                ])),
                Felt::new(BigInteger384([
                    0x50ccce4b8eb7d3a7,
                    0xf07873df350ec118,
                    0xf5328c7c5eaa8e56,
                    0x346b2f7cab233a69,
                    0xc7334d1348b7ad27,
                    0x0a2f3b3abc25dd49,
                ])),
                Felt::new(BigInteger384([
                    0x7873aac66d6e0dda,
                    0x810012785dc91689,
                    0x8da4ca6bd5a280e9,
                    0x98f51a2595e32790,
                    0x778c7a772c7858a0,
                    0x0829b65fdfc815e3,
                ])),
                Felt::new(BigInteger384([
                    0x72b348e6dc6acf74,
                    0x2c86e0c8bba95356,
                    0x193b6a120b8e1e76,
                    0xe164af8f4090bf37,
                    0x56783bb32ed1c4b6,
                    0x138af84954722c8a,
                ])),
                Felt::new(BigInteger384([
                    0xcee21adc9dd43bb8,
                    0xbe8a7b1e46be2810,
                    0x689031b5d36c7a56,
                    0x7b64349756ccc5c0,
                    0xdfc772c423abbdc3,
                    0x00a715fdbe757cc1,
                ])),
                Felt::new(BigInteger384([
                    0x2fff9b18ae34e553,
                    0x629a25a6cca00bda,
                    0x8820ad3ea3f9437f,
                    0x2bfd37250a6bac76,
                    0xe9c088ec686b5550,
                    0x179a8220cc30de83,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa0946263409f96a2,
                    0x5c82613f0fab8f89,
                    0xacad2b9537bb76d3,
                    0x3325ece2886a8beb,
                    0xc0ac92f077b3025e,
                    0x0f6940f92571f0b3,
                ])),
                Felt::new(BigInteger384([
                    0x53f28984ef7875c2,
                    0x3f54889c70dc07a4,
                    0x62191c9ec482ec60,
                    0x11e809224f04a9d6,
                    0xce0e709c1dbe3273,
                    0x02f114a4f03ea569,
                ])),
                Felt::new(BigInteger384([
                    0x7b07c53ba08dec90,
                    0xee408d03a76fe15b,
                    0x0ac64ee9ab238062,
                    0x047990761b270355,
                    0x2527fa758a1be8e3,
                    0x14fc54301bfaad15,
                ])),
                Felt::new(BigInteger384([
                    0x4b187d0c38235dee,
                    0x100618b35f484020,
                    0xc5a0d9160c2a57f5,
                    0xeaccd4daf93c6a3e,
                    0xa29b262bd8274c43,
                    0x0d3cb3450e9b5f1b,
                ])),
                Felt::new(BigInteger384([
                    0x8676fdd0ba01e268,
                    0xb760d97d40d3b0b1,
                    0x2d0277078a8a2f38,
                    0xa26bf12ff7c82505,
                    0x2ecb545d7c19e902,
                    0x03328f90a86d9bcd,
                ])),
                Felt::new(BigInteger384([
                    0x07b3181dfeb5dea7,
                    0xcc08823a27018fe6,
                    0x3d5e274fff6ebab1,
                    0xf08915ed7ff7abed,
                    0xe5d3fc92d860aa45,
                    0x00693abb00269c60,
                ])),
                Felt::new(BigInteger384([
                    0x499677d869eac9c7,
                    0x5a226ffc5e133e0c,
                    0xf77f586e34b12152,
                    0xd13625ffdbf07906,
                    0xcafbbaad700dd4a3,
                    0x0c2bd8af7718d7f6,
                ])),
                Felt::new(BigInteger384([
                    0x26f47b5c5acff3ec,
                    0xfe0b90642aa7c1d9,
                    0xf3e236ad8958c6f9,
                    0x7ff2ea261b1af9c9,
                    0xacbc9b8ec36ecc12,
                    0x15d83ed81ce4a499,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc76033774957117f,
                    0xe08b46e757b0e15c,
                    0x114486d91b498cfc,
                    0xd2ce2a55f683b15a,
                    0x234e02227fa83ca4,
                    0x065a10a494f502b0,
                ])),
                Felt::new(BigInteger384([
                    0x9cafc3c12d01b76b,
                    0xec037d3bb8ef4897,
                    0xb97dd29371f88da0,
                    0x5dd1d9d5e79ead08,
                    0x0e4021e2325cd64e,
                    0x0a8691b5e4112851,
                ])),
                Felt::new(BigInteger384([
                    0x1d7ddc887f9ca018,
                    0xdbdba70a5b3676d1,
                    0x5ead63f45b0f900e,
                    0xe80be247be6ca66c,
                    0xc641db4f9b6f13fa,
                    0x13e0ff678fc53f2d,
                ])),
                Felt::new(BigInteger384([
                    0xc1d7214d958dfe8d,
                    0xf8c9b00f134a7f61,
                    0xa14bbfc046b27827,
                    0xf1716c5b25bf6d12,
                    0x4a126640fa3f429d,
                    0x18420bb752e26c39,
                ])),
                Felt::new(BigInteger384([
                    0x4c4994fbd1a9ee89,
                    0xf078dd21425e671a,
                    0x0cbf97fcf8c2144e,
                    0x9814af542a60b8b4,
                    0xce383f077a9d27d3,
                    0x15ce94f47c38d4d5,
                ])),
                Felt::new(BigInteger384([
                    0xc1d400c55fc45d65,
                    0xa5181e922ada4fb5,
                    0x6288c910a30f0a4e,
                    0xbe62852bc722a69b,
                    0xe81ce76c155b6862,
                    0x1821d6c6bcbc5f4a,
                ])),
                Felt::new(BigInteger384([
                    0xf50af91fdcd80961,
                    0xaad71a0bbc1810fa,
                    0xe8416f75ec64ad5c,
                    0x25cf3c14f0aba932,
                    0xc0363a108ab77bc0,
                    0x03f2851602e63398,
                ])),
                Felt::new(BigInteger384([
                    0xf9581ed0e5ad0033,
                    0xc9a3e2da0fa05d54,
                    0xffcb8d1093d926c8,
                    0x9644902c8959d29b,
                    0x468fcaf0aab4571e,
                    0x0ba9410f5b8a5c12,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x8a93891fa2565ac1,
                    0x5bb06b9639858b6a,
                    0xb1a3f4b254f75496,
                    0x05be98f9523e4283,
                    0x65c81a19e8c3668f,
                    0x0c10526d526480a7,
                ])),
                Felt::new(BigInteger384([
                    0xd730aad989e788e3,
                    0x09f5c76f9f589507,
                    0x03700466c1525a17,
                    0x0b6a7eebba6ab5d6,
                    0x61ceccc52839611f,
                    0x1839e0b49f2e2700,
                ])),
                Felt::new(BigInteger384([
                    0x1451ee84543c74a4,
                    0x9cd11384a6fc8b30,
                    0x06dcbed0e11e4958,
                    0x343888df22f4d9d0,
                    0x50efd92598707f8e,
                    0x04a4c8cc61660628,
                ])),
                Felt::new(BigInteger384([
                    0x552fd06a7b35bc6f,
                    0x5b8463d82e419ce2,
                    0x2386742440c7eab6,
                    0x530d555c26387f73,
                    0xe76fc4a1a912fd31,
                    0x18f895520611ed3a,
                ])),
                Felt::new(BigInteger384([
                    0xda98d761515b827b,
                    0x7413ce3f4acf26c0,
                    0x9b60671deefe9eef,
                    0x5804a40474beea3b,
                    0xd24d802ff922d089,
                    0x1249bb10d2b82cc5,
                ])),
                Felt::new(BigInteger384([
                    0xd158b66a974c498a,
                    0xf518bb525d189f7f,
                    0xef901d1812501327,
                    0x282983c33a1b9999,
                    0x04b8748c0f72a34a,
                    0x0bdda2dc0fcdf1f6,
                ])),
                Felt::new(BigInteger384([
                    0x7f08369ee3dc442a,
                    0xb0cc4eb2207d35ed,
                    0x27790b5c538038e5,
                    0x30840e921728f8f4,
                    0x559ba875bb8736a5,
                    0x01121a75110f6307,
                ])),
                Felt::new(BigInteger384([
                    0x6ce33599d122322d,
                    0x2c0a740458ad6a53,
                    0x5be12369e8b437da,
                    0x27c2223ca0836485,
                    0x4ddee1fd681da69b,
                    0x0e95c5da56287be0,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xce152dce069605f2,
                    0xc02ece3b7aeb9042,
                    0x2d6a8002e6e75a30,
                    0x01f015addbb73e23,
                    0xc5893ed8aa7ba26d,
                    0x0b1a78ee60eb46e8,
                ])),
                Felt::new(BigInteger384([
                    0x061c5258bb0a4b5f,
                    0xd023e1d86b8520a3,
                    0x2752ab574c46427e,
                    0x7850a782b4ed90ff,
                    0x9490c5203a117708,
                    0x18f23abebb2a65b3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa6c0829fe70ec2bd,
                    0xd2b0620b8c8c0bf1,
                    0xf380ac3597626f34,
                    0x469b1bb1ce8e40ab,
                    0x3054b2a2e2a710b1,
                    0x05f52b216baee5f6,
                ])),
                Felt::new(BigInteger384([
                    0xa7366ef0358228a9,
                    0x09095c00a8edf3eb,
                    0x9e6bc0cb60c67794,
                    0x34c4ac2ea04510eb,
                    0x41a48e7d2622b925,
                    0x13b6a192e72b1954,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x8cdb481d9a1da89d,
                    0x411f05cf13601718,
                    0xc7d278d75aa6f317,
                    0x223e7baa94b687e1,
                    0x8fac09c51b471d24,
                    0x118d1d47ad7ae22f,
                ])),
                Felt::new(BigInteger384([
                    0xa077628704bb06ce,
                    0x1504db07d2f7669a,
                    0xef10e6c6263824b4,
                    0xb5eaf3c5f362c186,
                    0x67ed433ce226c602,
                    0x08f6b268b7a6f107,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa266375c8d271547,
                    0xa0623e5bf9235578,
                    0xe8695d21670bec72,
                    0x34d5ee385a22bb73,
                    0x666c7078c3486c40,
                    0x048379ad1c9d5a42,
                ])),
                Felt::new(BigInteger384([
                    0xf8e5729f424486ae,
                    0x7a17c135a8fb7ff8,
                    0x1bbba3e41f1c5ae7,
                    0x6ab02c498053016c,
                    0x81d5a0a9c97db757,
                    0x0484ca7ca873d703,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0a2b539c14a0b78a,
                    0x284adf8f56913cbd,
                    0x2faf807d850041b1,
                    0x275d36050557ab5e,
                    0x29634d27c04fc905,
                    0x08f967ea72f37272,
                ])),
                Felt::new(BigInteger384([
                    0xbbae786bc7965d78,
                    0xea8923f5354a8462,
                    0xdca88527af544629,
                    0x628043e8ddb0c07a,
                    0x4fbb61b94bd4a3e0,
                    0x196423c8b93ff6f4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4a4e9506de4e6c80,
                    0x7b99cf11e7ec40dc,
                    0x6c86866af0f097e3,
                    0x3620092ed6827afa,
                    0x6924c2d3cc7d9f7e,
                    0x19e5c5cc384abfb7,
                ])),
                Felt::new(BigInteger384([
                    0xff20f7f9ffe32a41,
                    0xedbe1885b09cf196,
                    0xb174f8987e1c1ce8,
                    0x1976ae4bbca49cdd,
                    0x6f2767054132113d,
                    0x184c706e8c4027db,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x8039982e9999361b,
                    0xc6f59223c792c548,
                    0x32b642ecfc35a565,
                    0x5e7e19db404b9a21,
                    0x048e0fb8ab8d828b,
                    0x10151547760afbff,
                ])),
                Felt::new(BigInteger384([
                    0x6fa15f1ea9502338,
                    0x9dd86960fa7f4682,
                    0xe017572a049a7b2d,
                    0xcb9393e65a23ed7e,
                    0xcebe22c2754c310f,
                    0x05fa35831a80a10a,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x809a2975a1a3bb8a,
                    0x83220b9818b68f93,
                    0x20dd989c81d9ff20,
                    0xfeaabe2ad4762ea0,
                    0x01f98699df3efe27,
                    0x1205250bb3eb7359,
                ])),
                Felt::new(BigInteger384([
                    0x2bea000e5d3c85bf,
                    0x7a63536d52ece786,
                    0x36fe962007b01466,
                    0x7ae1e32d59ce49fe,
                    0xae80b1d2df2a590a,
                    0x109f4a2ad8affef2,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x71a0bfc9f61b51d9,
                    0x52f3edfd6e30ea5b,
                    0x60fa3427ab7af630,
                    0x3b0af61a09ba5445,
                    0x382e9b2435e570a7,
                    0x195e933afd73e024,
                ])),
                Felt::new(BigInteger384([
                    0x7432c52351ce2c64,
                    0x7d3e645c5e41ec51,
                    0xf002fb7f436c86f3,
                    0x1756d1b3d6bf2fe5,
                    0x49d77228eb77f588,
                    0x1409df7ede99d2b1,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6435dc75dfacf4a2,
                    0xcd76a55ad93fb810,
                    0x80f0d27df75cb996,
                    0x1f81a2184e881337,
                    0x79495f893199e756,
                    0x168fb6c94178a550,
                ])),
                Felt::new(BigInteger384([
                    0x953cce91cdcaf705,
                    0x69e89fa056f820ab,
                    0x7d59df308129d07e,
                    0x52b4dfb1a563d92d,
                    0xb4d30335e20560db,
                    0x1192b14c87f6e05c,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
