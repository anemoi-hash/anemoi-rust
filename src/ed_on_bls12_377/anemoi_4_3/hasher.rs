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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x95d1dca7e7528b82,
                0x90cc5961e7dab214,
                0xa94a1f14d997cfb5,
                0x034c9dbe25eedcf3,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xbf365adbed3e1a1c,
                    0x0b689d8c0bf2bd1d,
                    0xf4c922e9ab9a04e6,
                    0x1299678c4eaf1036,
                ])),
                Felt::new(BigInteger256([
                    0x7b121a858f4ea0dc,
                    0xd644bdd17d7efb38,
                    0x106b5fbff674d2ac,
                    0x03b0065b18a57f2c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x489b62bf9e225b94,
                    0x5297a443927ea195,
                    0xd9991b317f0877a5,
                    0x012e7b547dc87c90,
                ])),
                Felt::new(BigInteger256([
                    0x50416da2327a6b7d,
                    0xb54cf3e2193b615a,
                    0xa7097c646579cd5d,
                    0x0ab44052faf979e5,
                ])),
                Felt::new(BigInteger256([
                    0xdafd255eba987ca1,
                    0x0dada83a9cb59b5c,
                    0xbced9147fd8a405a,
                    0x0622c050bba8c76c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd829ae2ef49b6179,
                    0xe5de862190a88eb7,
                    0xb4c41630251d6e9b,
                    0x0b44203a18dc04d7,
                ])),
                Felt::new(BigInteger256([
                    0x4d0c17b67670d679,
                    0x6d9f285c2a058d26,
                    0xb349ccf451bed34d,
                    0x02fd020a6fa0b02a,
                ])),
                Felt::new(BigInteger256([
                    0x1d9bd360fbf75906,
                    0x92f121f0c43fd39f,
                    0xc58e64b3b6d17328,
                    0x05d03c6a769eb7d9,
                ])),
                Felt::new(BigInteger256([
                    0xe9fa6fd032ae4d4f,
                    0x7fa50367cf67d5a9,
                    0xea7472a1c7a63868,
                    0x01ec6729f58c8462,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x05c8e9351e038754,
                    0x42b9cde58576cf14,
                    0xfb0577bb36b7ba31,
                    0x05575ca931db90a9,
                ])),
                Felt::new(BigInteger256([
                    0x37a0944b941ee8eb,
                    0xc0d2ab56297c01e3,
                    0x66918c50df6e8486,
                    0x0b6267171556cb83,
                ])),
                Felt::new(BigInteger256([
                    0x52659101d9bcd4be,
                    0xb248b541e93089a7,
                    0x86aa4e606ab26ab4,
                    0x0c778e4a7798f6e7,
                ])),
                Felt::new(BigInteger256([
                    0x79bb89ee3e89cf60,
                    0x3b73645207ed6da0,
                    0xe6865dd20bd2682a,
                    0x08ec815960478ed6,
                ])),
                Felt::new(BigInteger256([
                    0xdc957825952911b0,
                    0xa742e2eef3b214cf,
                    0x16f90ff15bd9c73c,
                    0x00f3aaad6a09cfec,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x11005ddbe87c1fbd,
                    0xf73e4329ba069a1a,
                    0xfa0a0c174b02091d,
                    0x062e033c598daba1,
                ])),
                Felt::new(BigInteger256([
                    0xda1507983779ffec,
                    0x6eeb6bc164ad48e3,
                    0x5702476804a65382,
                    0x035b4d418e83f874,
                ])),
                Felt::new(BigInteger256([
                    0xa47d43eeb246538e,
                    0x6818dbbeb3bc5c6a,
                    0xab119c0866ff3f13,
                    0x089666c06d974827,
                ])),
                Felt::new(BigInteger256([
                    0x27a2ff78f0505884,
                    0xf471a2a0aa3ac8b8,
                    0x0971a10ae94b1cbc,
                    0x1181ea6dd08ee4b6,
                ])),
                Felt::new(BigInteger256([
                    0xd9afc553f4ee229b,
                    0x03f04f1cc5306d81,
                    0xfffa86ff4753be14,
                    0x04da77f02fd77ce8,
                ])),
                Felt::new(BigInteger256([
                    0x2151bd44ba2ac8f0,
                    0xb28b168f5f3347a2,
                    0x3ad4a9bceefd0f3d,
                    0x0894995cfcd6509a,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x6fb220f63c62f845,
                0x1d6f4271ce9c857e,
                0x6ad0833f55da03b1,
                0x0c3f4648b06f6f50,
            ]))],
            [Felt::new(BigInteger256([
                0xa368761721e38126,
                0xbcf368b442e2977a,
                0x2feb58f7b6c8fed8,
                0x106a4c64164df1ae,
            ]))],
            [Felt::new(BigInteger256([
                0x4604826ff6ac50a2,
                0xdd52734233523773,
                0x1500e5df4320c0fa,
                0x0b7db8fcfa7cda0b,
            ]))],
            [Felt::new(BigInteger256([
                0x7f91abf44709c965,
                0xebeb8d92b0e5e770,
                0x2668f2933d2ef53e,
                0x06d23e597866642b,
            ]))],
            [Felt::new(BigInteger256([
                0x1165320b4e5ae3c5,
                0x10333b7bbb1d1bfd,
                0x60415b6d17877cf9,
                0x0487af4db7cae76d,
            ]))],
            [Felt::new(BigInteger256([
                0xb542777718f885da,
                0x10aa9de55c2ecf76,
                0xa792220169c63cb9,
                0x07fbe8e5785710c2,
            ]))],
            [Felt::new(BigInteger256([
                0x3cdac40bbf6df828,
                0xc152779e03db85de,
                0x6d1b53c118e0eb17,
                0x00d702747ac5b0d7,
            ]))],
            [Felt::new(BigInteger256([
                0x95ced302b6e11d7a,
                0x4f93555b60d8e004,
                0x569d60410334534d,
                0x029f4517c41f3ef1,
            ]))],
            [Felt::new(BigInteger256([
                0x3985cca5075068d9,
                0x7ed4babc8d70fe67,
                0x89963562ab4418fa,
                0x06847b8bad60cbc2,
            ]))],
            [Felt::new(BigInteger256([
                0xfb577e745fa56a60,
                0x315e0df0a6aec08e,
                0x16945b914a048e9f,
                0x0b6551bed3635fc2,
            ]))],
        ];

        for (index, (input, expected)) in input_data.iter().zip(output_data).enumerate() {
            println!("{:?}", index);
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0xdcec1ab70535dfd4,
                    0xdaa7fb183cfe62b6,
                    0x30e932a13b1c3df0,
                    0x07779c5778404650,
                ])),
                Felt::new(BigInteger256([
                    0x827d2976d3873f8c,
                    0x64da1a26c08ebdc7,
                    0x7e11eb4aa9a9d58b,
                    0x03be4171dc4b859e,
                ])),
                Felt::new(BigInteger256([
                    0x9ac8c9d6ad71e4bd,
                    0x43e508d9ff0b15d1,
                    0xaa3fd9486ef6307e,
                    0x09661a9aebe48a55,
                ])),
                Felt::new(BigInteger256([
                    0xb28f83aa8335eaa6,
                    0x5ac2daf65ffe0d3d,
                    0xf0b19680544bf011,
                    0x0f7382694b95bc67,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaaacc7dd2f9346a8,
                    0x1d2ac461d0ddc8a4,
                    0x00e485154816a524,
                    0x07feb6718623b933,
                ])),
                Felt::new(BigInteger256([
                    0x97daf6ab8551f2b7,
                    0x8e4222038a4cc46b,
                    0x7311696613fef8ac,
                    0x027e76379fffe659,
                ])),
                Felt::new(BigInteger256([
                    0x4f23ec1b5efc547f,
                    0xb1da73023871fe29,
                    0xd54451f36a812c44,
                    0x0f92748b3c8e2f1f,
                ])),
                Felt::new(BigInteger256([
                    0x090b4b1d806ea3e4,
                    0x586a62b4850a4c78,
                    0xe4ee669c2709e128,
                    0x0c0409fba76d0778,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8a5f54f7deeaf70d,
                    0x8b613acee58cd6bf,
                    0xe7263018bd53b267,
                    0x0500f24df9d6f5d5,
                ])),
                Felt::new(BigInteger256([
                    0x60420760e120f87a,
                    0x813150c6d87634a4,
                    0xec90176457e9daa1,
                    0x0068263a798a3480,
                ])),
                Felt::new(BigInteger256([
                    0xce931b948615e1da,
                    0xe8e588e4cc8516f2,
                    0x2458d55ccb761119,
                    0x009657727e91d809,
                ])),
                Felt::new(BigInteger256([
                    0x151b827eeee303b0,
                    0x00cdd0a3df43925e,
                    0xebecf56ec087b579,
                    0x0e0c2997d357aa6d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x33d87b1ae4306de6,
                    0x5e02cc42b872245a,
                    0x2268d2a4d0e0ab0f,
                    0x0969cf36596ecbb5,
                ])),
                Felt::new(BigInteger256([
                    0x29d48006573495e8,
                    0xde4487a482db7a49,
                    0xe29526a130dc73d6,
                    0x0b932e114735ae9d,
                ])),
                Felt::new(BigInteger256([
                    0xac25f7412efe9d86,
                    0xfdc604e131e22680,
                    0x56a08434e1db9ae6,
                    0x0fa3235318b489a1,
                ])),
                Felt::new(BigInteger256([
                    0x6f772350f1238f42,
                    0x273b0cb03fd75faf,
                    0x3238f98ba92101cd,
                    0x03c69b63a3c08fc8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe84aee9f702af044,
                    0xfaeba6df5a877004,
                    0x5bd25bd34a41461c,
                    0x03e9d4527103cd1b,
                ])),
                Felt::new(BigInteger256([
                    0xd494157f5380e053,
                    0xd3fd8f3d952a025d,
                    0x87cb4b944c7a2a5b,
                    0x0b3fcc2c9f93d23a,
                ])),
                Felt::new(BigInteger256([
                    0x206035f164395abe,
                    0x5f6e95b050fc65c2,
                    0xc2e561654cc1b1f1,
                    0x058639d6c3e30f04,
                ])),
                Felt::new(BigInteger256([
                    0x97360b969b5f05bb,
                    0xdda942bb400ab561,
                    0x06e43e94be2c45bc,
                    0x054e8003b42b0330,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc9e804357f90b737,
                    0xaa5046e54d59e635,
                    0x7bf129b309ba4d40,
                    0x09409138775be1df,
                ])),
                Felt::new(BigInteger256([
                    0xf07eef1926383a51,
                    0xdf635378afc2db6b,
                    0x3febb925b32cbd88,
                    0x1156f105c2fa3693,
                ])),
                Felt::new(BigInteger256([
                    0x01e664b6ec6b2af4,
                    0xa119731da3026631,
                    0x2f58248a0d026380,
                    0x1295004ad2f3911d,
                ])),
                Felt::new(BigInteger256([
                    0x835c6f3bb8e27775,
                    0x31843eed87bab304,
                    0x2676954a05419db8,
                    0x119b50ccb1cef09a,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x0f517e4f2523bbf5,
                    0x7d73200b630f1660,
                    0x1664a144154a7073,
                    0x0af24bb9e20af3f0,
                ])),
                Felt::new(BigInteger256([
                    0x9ccd535d362ed980,
                    0x8f1e6b4c0f337bce,
                    0xf419f74c223d756c,
                    0x002697f9b131afa9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc3353f14d8bf3b92,
                    0xf3022916cc6bccb2,
                    0x8f66880553c05211,
                    0x055edf302775cd3e,
                ])),
                Felt::new(BigInteger256([
                    0xcc69ac4ce1829265,
                    0x1259e8c1de8527b8,
                    0x3573fc955f6ba2c0,
                    0x06ef0f1e583d8b13,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x708844140b79fbe3,
                    0x013d40f3a23c4295,
                    0x5e31b894d92a02d4,
                    0x0fd83eab047eda82,
                ])),
                Felt::new(BigInteger256([
                    0x37c13826574a2fc1,
                    0x2ec148ce07f7cd4e,
                    0x54c3736d0ea0f5c6,
                    0x0e2b8b3dcbf7ba1f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcf3dccf21db4cd39,
                    0x1cf7f1791af867a0,
                    0xa3a32c09aaec5d4a,
                    0x10a3de51680d1c17,
                ])),
                Felt::new(BigInteger256([
                    0x5d82e0b9f3381ea1,
                    0xc6de0846966cc609,
                    0x2222773d660595ff,
                    0x106b9663d292e67c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb0031cd97831b298,
                    0x22e26d46ed3756c5,
                    0x165c35ca28b60a3c,
                    0x0dc64b7bd1be124a,
                ])),
                Felt::new(BigInteger256([
                    0x3406632e80595200,
                    0xf4f6abaf861e018f,
                    0x3a75361fa4205ac4,
                    0x006219068611233b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4db45b21ffbda129,
                    0xb5d37deeb022fe57,
                    0xe6002344bd728b19,
                    0x0381b9680374461e,
                ])),
                Felt::new(BigInteger256([
                    0xd47b734ccc1fa668,
                    0xbd640f5a6336e834,
                    0x14ec70505e58f790,
                    0x02b8cd53a4e81db7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4bbdeaf9fb5e1ff1,
                    0x93c584dea87a3f00,
                    0x808291e1728fd4ff,
                    0x0f221f150718f8fd,
                ])),
                Felt::new(BigInteger256([
                    0xda5cac8f424fb464,
                    0xfa5be630b1a414d1,
                    0xbcb97f1d99c3dc1e,
                    0x0cbe689299c07ef8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbbd689172165e3cc,
                    0xc60321b26cbd8336,
                    0x2ffec069e2f83397,
                    0x0eb7d4a3fc6aa939,
                ])),
                Felt::new(BigInteger256([
                    0x18f0ad59ec6da99f,
                    0x7d28601b4924c1c9,
                    0xc22be111a1bf9dd7,
                    0x0e3684895a5267c4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x317f448e4bfc8e4c,
                    0xb058e0af2d72ea0c,
                    0x8e79c6b66c241afc,
                    0x0aee9bd4e018bb61,
                ])),
                Felt::new(BigInteger256([
                    0x044593537ea796fe,
                    0xb7a917645c92dccb,
                    0x07c6b765cab59221,
                    0x02e68c6d524c3735,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd6a0e64cc2750690,
                    0x7eefbe2616cf4639,
                    0xd3c5c51e29704cab,
                    0x08a42cb72fbe0ea1,
                ])),
                Felt::new(BigInteger256([
                    0xbec22fd470c916ab,
                    0xadddf287a42d11fe,
                    0x042baf302fbb56b6,
                    0x126ca7dfbffd12e7,
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
            vec![
                Felt::new(BigInteger256([
                    0x2d5ff33671c16f54,
                    0x160465cd53e69b75,
                    0x6ad83c3b4a3a630f,
                    0x0696560a2daf81e1,
                ])),
                Felt::new(BigInteger256([
                    0xf90a888cdd0d46f2,
                    0xaa53c1bb63ee9601,
                    0x38a331dc845eae0c,
                    0x11ce400e1ed6f1cd,
                ])),
                Felt::new(BigInteger256([
                    0x8b0334b94194bff8,
                    0xad006ec80bf69cff,
                    0x39b87b190d2b3da9,
                    0x013ab149fc6c5eb6,
                ])),
                Felt::new(BigInteger256([
                    0x3cadb9bc1b7cd978,
                    0x288f5888d1ed2568,
                    0x7bbcfa2f20e65ae8,
                    0x005b556790cd5d07,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa91bed185f95eca2,
                    0xbc7dfc3f1d64d232,
                    0xfee16b3d2687fe75,
                    0x021945c74a4cfe6a,
                ])),
                Felt::new(BigInteger256([
                    0xb81a692663bbe171,
                    0x80336221f9e7afdc,
                    0xfed60e9f1fd9ff95,
                    0x0d1cf404fa0354c5,
                ])),
                Felt::new(BigInteger256([
                    0x198573c8ef59486c,
                    0x43c47d2c8f3d761f,
                    0x68385ddaad9da07c,
                    0x1034cbb5764a29dd,
                ])),
                Felt::new(BigInteger256([
                    0xb3a40854bf37f265,
                    0x65d86b4fb441e80f,
                    0x29ebb6c5f71c399f,
                    0x0e4cd723d640709f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x86330a02ec4baf4f,
                    0x03785a8ab644ca8b,
                    0xec5b4224086d0452,
                    0x05bfefd8e8656384,
                ])),
                Felt::new(BigInteger256([
                    0xaae2b522211d296f,
                    0x6ce4a5ff6bf4254f,
                    0xdced5b33d748d786,
                    0x0cdc67cf09cec588,
                ])),
                Felt::new(BigInteger256([
                    0x0cffe4b47f7e2ef0,
                    0xbd4b9898d1fa9eab,
                    0x3af260141493dea1,
                    0x02e829a63939e7df,
                ])),
                Felt::new(BigInteger256([
                    0xdf29889d137f9ee5,
                    0xb9d89d632e00f7e1,
                    0xeeb3e19f7045977b,
                    0x09a66494adedfdac,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1d1e6d303461e913,
                    0x591d69244658b95f,
                    0xe930362cfde473c8,
                    0x0fadee2a1c0bfd3a,
                ])),
                Felt::new(BigInteger256([
                    0x9922a384fe91ed23,
                    0x9ae4f437254f17f5,
                    0x5272e2e3693e71ef,
                    0x114369ba65ee64da,
                ])),
                Felt::new(BigInteger256([
                    0xd9e057265af21257,
                    0x719aea9e1854b95f,
                    0x05b34742726669f0,
                    0x126522b69f0e2a7c,
                ])),
                Felt::new(BigInteger256([
                    0x6192e79e5c4a8376,
                    0xa8e509f82c5dd35e,
                    0x734c9dd56aa4da83,
                    0x0f20222b317231db,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3aaa3cbdaf9ae848,
                    0x435f973221fc4da1,
                    0x1977956234533d95,
                    0x104f20f71a239447,
                ])),
                Felt::new(BigInteger256([
                    0x3e618c5ad7b52288,
                    0x79b04c485c9e1f3c,
                    0x496437b2a1ec1231,
                    0x0f4533cb723652f8,
                ])),
                Felt::new(BigInteger256([
                    0x360528b458fa12b3,
                    0x31dc75411f72c296,
                    0x2a5b14cac1040a7f,
                    0x0086e7d155a9457d,
                ])),
                Felt::new(BigInteger256([
                    0xc5a6679404194994,
                    0x21d86ec5db8c5756,
                    0xca3cb8296938144f,
                    0x08494cec3491a129,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x36b54a6662cbcd4a,
                    0x8761bac334e0299e,
                    0x6e080b46a5822b78,
                    0x03f17782d8257db9,
                ])),
                Felt::new(BigInteger256([
                    0x0f58ff74a7542103,
                    0xea19f2e6f6572c01,
                    0x2ebe09138ddf0b8d,
                    0x006695a9cd89a7f9,
                ])),
                Felt::new(BigInteger256([
                    0x0fce9668d7d71a1f,
                    0xc6a2feca6719e50f,
                    0xa376fd4a95f1e2b4,
                    0x0c94335d86315214,
                ])),
                Felt::new(BigInteger256([
                    0xf51a7bca517c1f92,
                    0x52e3b57140847838,
                    0xeb17e2e8dac9477f,
                    0x10bcb305c6d73885,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xac1ed1ac5b529575,
                0x0c918b577242922e,
                0x0a7e98903787e5e0,
                0x0b18e3b3933ca39a,
            ]))],
            [Felt::new(BigInteger256([
                0x8f9eeb61ba41cdf7,
                0x055c11d8aaf0f46b,
                0xc4da849ab32bf4d2,
                0x0c4dee4e7fb35851,
            ]))],
            [Felt::new(BigInteger256([
                0x9e37fc3a62c42ba3,
                0xd65412c2da340fe2,
                0x5240dee38b934898,
                0x0b58648a3649ef4b,
            ]))],
            [Felt::new(BigInteger256([
                0x22af2dac10ecebd9,
                0x8a2b82c0e1652da9,
                0x65115628b4ba4348,
                0x0e640f56a0735d3d,
            ]))],
            [Felt::new(BigInteger256([
                0xafc08c7a677009cb,
                0x82a509096b91adb2,
                0xd8a36807625674ef,
                0x0eed00c94de1ef35,
            ]))],
            [Felt::new(BigInteger256([
                0x5ae9f41aff3bb47b,
                0xbe1702e99d763ae9,
                0x0f9301ae16b4417f,
                0x05485209a17e0136,
            ]))],
            [Felt::new(BigInteger256([
                0x4309332b8fa79182,
                0xb57863b0e9b8a861,
                0x9a3de513bd5c238a,
                0x00293639d900de12,
            ]))],
            [Felt::new(BigInteger256([
                0x7e6db5b99a867d41,
                0x258632c603612f9d,
                0x2724041e8f1644ba,
                0x0bf85628b8b2d333,
            ]))],
            [Felt::new(BigInteger256([
                0xc0c8127120214cb9,
                0xc845b3217a748ccc,
                0x7835f56b05812564,
                0x0e29de67c5955fa6,
            ]))],
            [Felt::new(BigInteger256([
                0x27eb0a6f10fb78c4,
                0x5cef556edbd04808,
                0xc2ff7d9fa06766ee,
                0x0f5b35ff75b11dcf,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
