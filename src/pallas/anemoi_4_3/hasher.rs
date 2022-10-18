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

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0xdbd494ee493d3e7f,
                0xe7312d2418b104c8,
                0x302da55763d2487b,
                0x031b0d3d56cfa107,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x1f4e98578e767580,
                    0x83234c57ed1b185b,
                    0x0061642a82c7fe31,
                    0x1e44160833929985,
                ])),
                Felt::new(BigInteger256([
                    0x8adfc2c8ba2db9c3,
                    0xbbd4eb02ac447143,
                    0x5bf6685a8c5628de,
                    0x0cfb4c4bcdeeff82,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb31df3f65242e4aa,
                    0xe3a1be75ec56e680,
                    0x95ab9513e745e918,
                    0x30b68b283f5e949f,
                ])),
                Felt::new(BigInteger256([
                    0x6bf4f8d6854cf1bb,
                    0xdaf6ad33499e8015,
                    0x4eefbf0c3f8efa4b,
                    0x083b3122909c9dfc,
                ])),
                Felt::new(BigInteger256([
                    0x185dee3e889f8551,
                    0xd1ce481987ad10fa,
                    0x4616ec50be2e0d10,
                    0x2601a4f928c0b697,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x894b13a8fbb88d28,
                    0x5e5d2ce4993b409f,
                    0x41d2a80daf6e6bcb,
                    0x30d7bb380f8caa57,
                ])),
                Felt::new(BigInteger256([
                    0x7710e2e06ae670b0,
                    0x58989b728f2bc67e,
                    0x7b067a4c6fea9a97,
                    0x120680d77e7eef47,
                ])),
                Felt::new(BigInteger256([
                    0xd6528916941542d4,
                    0x8b8532ac7a940adb,
                    0x52abca29256e9aa6,
                    0x15968f52bb2cfbb9,
                ])),
                Felt::new(BigInteger256([
                    0x3d35f9a4015da7cf,
                    0xf58c31d31fa9e6b7,
                    0x95d78201c68b2445,
                    0x0f4690e9daf4610a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x44a8d52db266c34a,
                    0x53c32dc14f9c7270,
                    0x9efb385a6a7efa95,
                    0x2364dc1295552f7e,
                ])),
                Felt::new(BigInteger256([
                    0x8fb577389633160c,
                    0xa4a35dfdb070b024,
                    0xd9b4e4baa587b338,
                    0x38aea43da5596085,
                ])),
                Felt::new(BigInteger256([
                    0x5158c86ef3f39845,
                    0xda4cb6ee6b25e597,
                    0x8ee0858806284075,
                    0x14b06337c741798e,
                ])),
                Felt::new(BigInteger256([
                    0x8adab84deab5e616,
                    0xbbfb65868f283a1e,
                    0x8d290cfdf43e7bcd,
                    0x2aafd922215b6fbe,
                ])),
                Felt::new(BigInteger256([
                    0x2bf52021042d5a78,
                    0xf0c7312a887bbf52,
                    0x90d599dfa886603e,
                    0x234bc3bfce5311ed,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5c857469adffd31d,
                    0x20eb68beda91bf43,
                    0x279a58482f9727c4,
                    0x131909529966433c,
                ])),
                Felt::new(BigInteger256([
                    0xc5b02004dda18436,
                    0xb3731a7ff53f2cf5,
                    0x5ec8e359f1a3c295,
                    0x22f84d244c7e8065,
                ])),
                Felt::new(BigInteger256([
                    0xb9051be8997a4847,
                    0xa2f4494f5ba6e503,
                    0xe2b7bbcba6eaf4f7,
                    0x2dc6576844be6b75,
                ])),
                Felt::new(BigInteger256([
                    0xecff8c1128d7b73f,
                    0xeec2975668ca284e,
                    0x7faed76f44cd9773,
                    0x322680e5a359d09b,
                ])),
                Felt::new(BigInteger256([
                    0xc2e1953558093445,
                    0x9c37f533ef3a510a,
                    0x1a1e46a090597ff7,
                    0x1a2f67d2cc9f2164,
                ])),
                Felt::new(BigInteger256([
                    0x26ea8fd9b0913306,
                    0x1432c546bef8b400,
                    0xf9f4fe842d7edd96,
                    0x148c071747ca31ee,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x344583ae65d29f8c,
                0xcd9586658d9e8a43,
                0xa397bd8898de5d21,
                0x3ebff10271a29362,
            ]))],
            [Felt::new(BigInteger256([
                0x09cb1c2733c0341f,
                0x0128e71887c51daa,
                0x7fd9c631bbe40d38,
                0x22316d317e65df99,
            ]))],
            [Felt::new(BigInteger256([
                0x1803b6ed734ab282,
                0xa68dcb1500c77ef8,
                0x5de7f0b1612ece9e,
                0x02fff9ba1d6a6989,
            ]))],
            [Felt::new(BigInteger256([
                0x8dc6c97386f8a7ba,
                0xa385b36e8e48f116,
                0x59d733a93b213349,
                0x3a159e33aab73325,
            ]))],
            [Felt::new(BigInteger256([
                0xc9038d9f6d720597,
                0xff498e2779858c25,
                0xddc3cc7bd8818dd2,
                0x08f25fdb65838100,
            ]))],
            [Felt::new(BigInteger256([
                0xa6cda08a3ef9144b,
                0x3e1eb405bb49b5df,
                0x3446122d34cce0fd,
                0x2eed2e2b498bc500,
            ]))],
            [Felt::new(BigInteger256([
                0xca86416cf50a7a05,
                0x12f31e6c0d8459e2,
                0x54f6bea282ceb68b,
                0x3c46c9d40e27055e,
            ]))],
            [Felt::new(BigInteger256([
                0x7174f24439b62c2e,
                0x942439c06ade5c08,
                0x9fc3509ce49b4f34,
                0x28c860f0ed54fab5,
            ]))],
            [Felt::new(BigInteger256([
                0xb248c7a613ead93a,
                0x93b53f188b8eb859,
                0x39e10f7970127de4,
                0x02d0f9fec70b075f,
            ]))],
            [Felt::new(BigInteger256([
                0xb6fad0df00128e29,
                0x79272f7ff502fd5b,
                0x66c32ff683a89b1d,
                0x13f1e7e133037f4c,
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
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x5875797c73fd33b6,
                    0x34213e793bee56c3,
                    0x560cdb16ca3d1433,
                    0x1bc01cc16fa186e2,
                ])),
                Felt::new(BigInteger256([
                    0xdc4672e0dd32f08b,
                    0xfc8da870bccaf74e,
                    0x613728419a414805,
                    0x39a6a13a0edee0f8,
                ])),
                Felt::new(BigInteger256([
                    0xedf573559f1fc9bc,
                    0x5888498c285232a6,
                    0x88d51054acd304c1,
                    0x35580c76dc683383,
                ])),
                Felt::new(BigInteger256([
                    0xde6b3662cbeb6e44,
                    0x027f891ef34b521f,
                    0xce369aa893006a72,
                    0x0d50f9b805e5eac6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4f9efbac29d4860e,
                    0xa431eddb87abde11,
                    0xb6646ed4402cffb2,
                    0x0a6aaf4f8ddc89cd,
                ])),
                Felt::new(BigInteger256([
                    0x9b6b943af356a42e,
                    0x573eb6142652919b,
                    0x9df8db5e57d0ecba,
                    0x004a0defed0f77dc,
                ])),
                Felt::new(BigInteger256([
                    0xf0c386f254977252,
                    0x52b232d6d054e0a7,
                    0xd7da716cf3eea71c,
                    0x0d98cd5eff4e90b3,
                ])),
                Felt::new(BigInteger256([
                    0xd8b70c46f551f64b,
                    0x757d3ac91579242e,
                    0x04028ce67861eb94,
                    0x2e51774246d378c5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x21a0cb9503568f11,
                    0xafff8b7f4194449a,
                    0xff4f3cef4acad7ad,
                    0x1e6de8cca47aa2a4,
                ])),
                Felt::new(BigInteger256([
                    0x945851861bd9cf2d,
                    0x344962db23120338,
                    0x596a4d8c4681b5ea,
                    0x05b6d9062183e52c,
                ])),
                Felt::new(BigInteger256([
                    0x0f0f4cfadba16b10,
                    0xb5f1a3424fcf7283,
                    0x9a0334844f630052,
                    0x0683163eb5e8696c,
                ])),
                Felt::new(BigInteger256([
                    0x94dd39c1d8be1575,
                    0xe2c69bf65d043b3e,
                    0x47a8c2331d44cc5c,
                    0x0fab2835bf3e581b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x084cb71c8c54212c,
                    0x9df8bdae670717ec,
                    0xe1cc08ba9a6a2b05,
                    0x238a39c63a0e3c07,
                ])),
                Felt::new(BigInteger256([
                    0xc7415f8827096d7e,
                    0x9f4e6cabc1199790,
                    0x61f7de8986418284,
                    0x196a9b0182290acd,
                ])),
                Felt::new(BigInteger256([
                    0xb293e021c03383a3,
                    0x64873d446b10cdbd,
                    0xd1cdbcd54f9e6139,
                    0x00820b4fd75b827b,
                ])),
                Felt::new(BigInteger256([
                    0x0091b98787ece870,
                    0x9ecdaad49791e764,
                    0x2c78a114e717f748,
                    0x0a16101e78e2dc25,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6131b85cdb5aa5d5,
                    0x9b0e0114c4eb30d0,
                    0x76b27ab1fedb4f88,
                    0x19492833867a7daf,
                ])),
                Felt::new(BigInteger256([
                    0x25aa87afd0821d46,
                    0xfd63a658f3550e6e,
                    0x0643985b6967dce3,
                    0x30e9bd7d0ac8913c,
                ])),
                Felt::new(BigInteger256([
                    0xc68bd8451cda5980,
                    0x877404ab4354e730,
                    0xd66af1c0313bb32e,
                    0x2c844311c95f17d9,
                ])),
                Felt::new(BigInteger256([
                    0xe5fb471c52be44fc,
                    0x4a3a640d119b351c,
                    0x9fc75e699a7a0b7d,
                    0x2c0fe63b6cd318a0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcf4246485eecf2cf,
                    0xf8b2674aa0167c08,
                    0x030e4f0f8a0bd104,
                    0x061b14af18d4c436,
                ])),
                Felt::new(BigInteger256([
                    0xf85374e9070cc75d,
                    0x25442fa2916a7d3a,
                    0x931aa78d36c95f35,
                    0x328e5513ae319d2d,
                ])),
                Felt::new(BigInteger256([
                    0xbdaf61a2dc93683a,
                    0x916d76ab57956514,
                    0x04b71e080debce95,
                    0x27a70ccf52167d7b,
                ])),
                Felt::new(BigInteger256([
                    0x628c07d187b64035,
                    0xe1c74ac4d89d5aec,
                    0xbc6f068e4543380c,
                    0x0a3495011c8f103d,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x2082e818dacfb858,
                    0x42cebca5f7626130,
                    0x192807eb5ca01f30,
                    0x07a537d5ee9d6e7b,
                ])),
                Felt::new(BigInteger256([
                    0xfa2e989438c33685,
                    0xc45a229b0cb51738,
                    0x55bba04d7669c6a2,
                    0x09096504088d7405,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf82f777c01d8d248,
                    0x99ea64dec1cc9947,
                    0xfdbe1a2a53df5582,
                    0x0803c16211774e46,
                ])),
                Felt::new(BigInteger256([
                    0xa66c433d53f93250,
                    0xd92a699fd6ca3da9,
                    0xd40725e33145a4c0,
                    0x2d772119bf1a253d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc672ba6718d79261,
                    0x8f28dc68babfde60,
                    0x702d5875aeee2063,
                    0x303bc9aea89b83df,
                ])),
                Felt::new(BigInteger256([
                    0x50e3ec9aa916acc8,
                    0x8e4a4141333964b1,
                    0xd9682eeb2505ffde,
                    0x26982ed50c1c84c3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8233e87fb32d7164,
                    0x1b288fd16df24f75,
                    0xd264c52369b73fc8,
                    0x38d742ebfce78643,
                ])),
                Felt::new(BigInteger256([
                    0x8e64d849bcdb4cc3,
                    0x3c6b851f46c9c046,
                    0xb45168fdc885e258,
                    0x321d3a5807171718,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa8988ec6a99b9117,
                    0x768d61f08903e526,
                    0x08aca87d1df7418f,
                    0x10ca7a3a51d59f82,
                ])),
                Felt::new(BigInteger256([
                    0xcc78e288b130d586,
                    0x120934bf2a21088d,
                    0xa72b3030ca6b2ad3,
                    0x2557323e02447b3f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbb7f7d51b1a8ec7c,
                    0x366dac03318daa4b,
                    0x4a9f9cfa0ab62926,
                    0x2c0eca5aca55bd2b,
                ])),
                Felt::new(BigInteger256([
                    0xe94d0330db824b49,
                    0xf57f2508465910e6,
                    0xea64193976552a83,
                    0x29049c8f87ac0044,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x50e84c142aca0416,
                    0x84501615afeac9aa,
                    0x05cc9ae05c6a1472,
                    0x206cbd8c7960bbd8,
                ])),
                Felt::new(BigInteger256([
                    0x93d522ef11af34b8,
                    0x6a6450e837e2cae6,
                    0xfbca8f5730945878,
                    0x2488437950c424f2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc86c193e17131959,
                    0x1ed69ffa2c60fc1a,
                    0x0f469f7b4ad43e13,
                    0x2b3fa233156e633e,
                ])),
                Felt::new(BigInteger256([
                    0xb25b93b16f073684,
                    0x4810c5eaf2263fb7,
                    0x5e7804bcaaa0874d,
                    0x101d6f0ec91476b9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x572bc2924d93789c,
                    0xda27aac9a66b5b5c,
                    0xa8057b6aac3c2d19,
                    0x3e91d86288bc4893,
                ])),
                Felt::new(BigInteger256([
                    0x4d1861cce89c3686,
                    0xe4f2877713393c88,
                    0x370f2b9088d1aedc,
                    0x3c4cc23cec432c45,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x072523fecbaee754,
                    0x3d5fd6ad9f2bd95b,
                    0x75988936581b159b,
                    0x3829f84d386e320f,
                ])),
                Felt::new(BigInteger256([
                    0x743e6189515df714,
                    0x0dd75d3e9078df2f,
                    0x419c9d0e5644e9eb,
                    0x3f8e72a55fbc17f4,
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
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0xd2073603f4ee957f,
                    0x21163a0914795b1a,
                    0xd8cba74775712936,
                    0x30799de47ab7f060,
                ])),
                Felt::new(BigInteger256([
                    0x6a60930dd6186dcb,
                    0xd7ecc8b13e02f2c7,
                    0x6507052821ef3c46,
                    0x0c262415bb298926,
                ])),
                Felt::new(BigInteger256([
                    0xb356d4b40a4ba855,
                    0xd37507109f6bb16a,
                    0xa2a6230529f10b9a,
                    0x3115d5a876f58dd3,
                ])),
                Felt::new(BigInteger256([
                    0xd732b6f26b31c985,
                    0x679a07341b573575,
                    0x606c9da64ee5b3df,
                    0x32dd4644aa87bc93,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbc80049e939c8da3,
                    0x2487ffe1b5b1749a,
                    0x60439cc15b5e019b,
                    0x3225c7a48daceba9,
                ])),
                Felt::new(BigInteger256([
                    0x42367d6be4433144,
                    0xbe2a5f4664e4b8d7,
                    0x7586069a7978cadd,
                    0x099de7747ec295dc,
                ])),
                Felt::new(BigInteger256([
                    0x882da5808a983e88,
                    0xc1ca2d55ecc21cf5,
                    0x3022d3cf3fd70dac,
                    0x12b8c770758fc58f,
                ])),
                Felt::new(BigInteger256([
                    0x0ec1f01374f977a3,
                    0x722ac0f868954bb7,
                    0x5cf15abca1c3ecdf,
                    0x248c989fe7904b76,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6034d9ee0ef058c0,
                    0xed9e2af5c32dc5cd,
                    0xd5e152d0b81fb2e8,
                    0x32ef41d06ecd937a,
                ])),
                Felt::new(BigInteger256([
                    0xc2ba0cf15b75130c,
                    0xb93ff2d85f363d03,
                    0x514c20af55e63e44,
                    0x1b05ba8c6394b144,
                ])),
                Felt::new(BigInteger256([
                    0x57769093ffa160cf,
                    0x4fbd90e674979449,
                    0x258174afe4bace48,
                    0x053f265e7ab7b67c,
                ])),
                Felt::new(BigInteger256([
                    0x7ce46e84d8de2df8,
                    0xd92605d23cbb157f,
                    0x2e1f87e280bc9d2b,
                    0x22a690cd63cf8324,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x050fa020e4005e01,
                    0x094ad41614836c28,
                    0xab7c6b6b2e5c41b2,
                    0x2e0d907ad520e9f5,
                ])),
                Felt::new(BigInteger256([
                    0x4c2754e89c59e564,
                    0xe1ec99358d98f7cb,
                    0xb7b7268a7e4218bd,
                    0x1f17fb979682bbe3,
                ])),
                Felt::new(BigInteger256([
                    0xa50c8aa743e331c9,
                    0xe8b20ccc6274d512,
                    0x994a04465e7a8230,
                    0x0b86d036db62ad6f,
                ])),
                Felt::new(BigInteger256([
                    0x06deda5450cb9d7b,
                    0xc4e6bff6cd3a727c,
                    0xc1787c84b4d0b741,
                    0x256f0e3a44506035,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1fc0e6deb6d49aff,
                    0xef14c2e0ce5b37da,
                    0x04582bfd1011cc87,
                    0x230bdbb1dc0d0122,
                ])),
                Felt::new(BigInteger256([
                    0x5536a8242ac34099,
                    0x8a1ede52be5e4bdf,
                    0x5ca100008832ab59,
                    0x34a3656e7b36f27c,
                ])),
                Felt::new(BigInteger256([
                    0x34592d77f0c3dc63,
                    0xfbf6cd1cace0c92d,
                    0xa89cfc5f585c769f,
                    0x05b4cff5e45f3d13,
                ])),
                Felt::new(BigInteger256([
                    0xc352d96b1879793f,
                    0x6dfa58892ec17248,
                    0x86ac146065efe7bb,
                    0x1a40a9d70dace2a9,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe55f9f7d28184f2b,
                    0xaee12e04dca400e8,
                    0xe5b2770fd96ac703,
                    0x271f5c9ff1b627c3,
                ])),
                Felt::new(BigInteger256([
                    0xf5cb2bd0a70e2d80,
                    0x6c3ef5a24a565d1c,
                    0x89fbd49d8cd41a03,
                    0x35ddd0b05bf763cb,
                ])),
                Felt::new(BigInteger256([
                    0xcb7ae1620208ed22,
                    0x72d5ef7e8e9c75ab,
                    0x7aa348bc4a19ea21,
                    0x0269d52b58319c2a,
                ])),
                Felt::new(BigInteger256([
                    0x6c5a02917a5f0f9c,
                    0x6c8f5b9d9a3ed147,
                    0x26c3ffbf94d7ca1f,
                    0x1b8d5be6c718e9a7,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x1ab180ad1392eedd,
                0x0728df4104177869,
                0x6ee3a838d309e5d3,
                0x10ae9cd9f72ae280,
            ]))],
            [Felt::new(BigInteger256([
                0x9e9bbab955d20498,
                0x7314ce7e9896d6f1,
                0xd1c5400d8524fa43,
                0x357ae27bd0917384,
            ]))],
            [Felt::new(BigInteger256([
                0x7e297614c1ee3f28,
                0xfb2c84ade4ac49f6,
                0x49958760d3f42041,
                0x16d3f883b4b808a3,
            ]))],
            [Felt::new(BigInteger256([
                0x776b8fdc7008be26,
                0x354d7bf4ab6f16a0,
                0x86b62e21323d2220,
                0x2af47d4403fe9d5c,
            ]))],
            [Felt::new(BigInteger256([
                0x7a2637bf73ef97a6,
                0x6b85856060395b30,
                0xee8b3014dff61642,
                0x2e57ade0d7b4be00,
            ]))],
            [Felt::new(BigInteger256([
                0x0fe11e1771a2a4f9,
                0x8820263f7700679b,
                0xf15052f6126d1a85,
                0x041f9ff480a43f59,
            ]))],
            [Felt::new(BigInteger256([
                0xebe12a6292d3b887,
                0x3365644d7b33825e,
                0x5a23b103bd3e1f3e,
                0x34fa64f25ebc9a2c,
            ]))],
            [Felt::new(BigInteger256([
                0x24139c17c2c51369,
                0x17d864313880c532,
                0x22cb2aa23f520224,
                0x384e177a8585e3ac,
            ]))],
            [Felt::new(BigInteger256([
                0x7496c0e34d292521,
                0x427beeb62f38e808,
                0x0bb5d5666d912052,
                0x0a0fef8ebfcc4f82,
            ]))],
            [Felt::new(BigInteger256([
                0x80c86e5b94648e56,
                0x8a6d45b1826b4e57,
                0x0d757bf433d3fb3d,
                0x3d603ea782624a13,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
