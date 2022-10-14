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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x0bfdf7b8f404f19a,
                0xf138c5b1a277c376,
                0x79d03b65b35ea5be,
                0x3a51667d7e0a634b,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x03cfea5e4bdc859f,
                    0x2c9c790ced70b6c3,
                    0x3edcf6788dc5a545,
                    0x2446c5c740fc22c4,
                ])),
                Felt::new(BigInteger256([
                    0x52e5bbb045cc76a2,
                    0x9a28535b9c752a88,
                    0x747e0ae49805af32,
                    0x2ca74b0af820291f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8fd1f439c01b844b,
                    0xb55805374bb89225,
                    0xcc30ebd9ebfffbc7,
                    0x300e71e585a7d536,
                ])),
                Felt::new(BigInteger256([
                    0xb7748b305014b72d,
                    0x99faf017263a4cd3,
                    0x6ac179e7a062782d,
                    0x3ce081fb1c56978f,
                ])),
                Felt::new(BigInteger256([
                    0x1931ab373d9e6e87,
                    0xb72026416ffcb5a9,
                    0x0bb0eaad7383e0c1,
                    0x2618ed68b4863ba6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x62ef67525e0a851f,
                    0xd71a16b37d4ffc4f,
                    0xd0e63a48da26a918,
                    0x1d7a2ae83d8d2871,
                ])),
                Felt::new(BigInteger256([
                    0xece51ab89e34e648,
                    0xce4498115528e7a3,
                    0xa5f5be2a8f3a8918,
                    0x28d1eaee7450081d,
                ])),
                Felt::new(BigInteger256([
                    0x155627f61893deae,
                    0xce9749d107530a07,
                    0xa725a7f3478479df,
                    0x047e4674a8611c5c,
                ])),
                Felt::new(BigInteger256([
                    0xd6d63501df45ad76,
                    0x6ae27857dbf38597,
                    0x2ceabe1151743888,
                    0x138ad521599c744e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf13f8836ca6abc15,
                    0x94a3016019d607d2,
                    0xcb1f1ca5d6f26348,
                    0x0f5fdc0d9dc9261d,
                ])),
                Felt::new(BigInteger256([
                    0xdbfa54dd9f1c9e54,
                    0xe4189e7549db20d1,
                    0x497237ea24026f28,
                    0x294bcd4f59942c2a,
                ])),
                Felt::new(BigInteger256([
                    0x2915ac0096998af3,
                    0x487bad5c61636be8,
                    0x8ad81c2c5c81c0ab,
                    0x37c9a976827cbdd4,
                ])),
                Felt::new(BigInteger256([
                    0x43ddea51b3801f00,
                    0x4997887f5420eb1f,
                    0xece210a060b227cb,
                    0x3c2ba2ed7ae7b4f4,
                ])),
                Felt::new(BigInteger256([
                    0x2898e782d15c4566,
                    0xd1e944c26482e26f,
                    0x92d80fa1f17fc380,
                    0x3e249eec02165a73,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x12b034aa0114cde3,
                    0x86ffe8757e677ec1,
                    0x9330dae8e98a83ca,
                    0x318c11eb72788537,
                ])),
                Felt::new(BigInteger256([
                    0x4bc9e685f40d7c50,
                    0xf64dd4e585d3add1,
                    0x12b168c12557976a,
                    0x340217f3b1ad44b7,
                ])),
                Felt::new(BigInteger256([
                    0x6a656c4b172221b9,
                    0xede1e496fb83fcad,
                    0x9ab26adce6d74765,
                    0x26b5f61722668d79,
                ])),
                Felt::new(BigInteger256([
                    0xda3264625a9b96f9,
                    0xcce48fa213905978,
                    0x09c012a14d4fe1f4,
                    0x3d2e3a1718b27752,
                ])),
                Felt::new(BigInteger256([
                    0x78802bff111f10f9,
                    0x51c79ed1db9f9adf,
                    0xd349a4788d02dd9d,
                    0x292668b5012c1584,
                ])),
                Felt::new(BigInteger256([
                    0x51f1b70ff77d1e8a,
                    0x0729108fa7ce2ab2,
                    0xdd39a930be0c9fae,
                    0x19cffe05655618ff,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x52d75419594fae38,
                0xc65727cca069eeca,
                0xaf6ccbb0f9d731ac,
                0x18a29675be7d4182,
            ]))],
            [Felt::new(BigInteger256([
                0x442a7c4f7be2e47c,
                0xc42555b92241822d,
                0xf96f857dbf4307da,
                0x16827efe783b8ca3,
            ]))],
            [Felt::new(BigInteger256([
                0x93ebb1c76a951afe,
                0xdfeee93e07f3021c,
                0x68ede8c01866ac5c,
                0x1ba4c1b0809ed802,
            ]))],
            [Felt::new(BigInteger256([
                0xd5a52781610cb7b4,
                0xcca0962fc4d57521,
                0x78e10961d0496049,
                0x2069fd65e315a752,
            ]))],
            [Felt::new(BigInteger256([
                0xa28ec172b160eb30,
                0x9b02fafd4c9ab62c,
                0xc485dff986b3bb9a,
                0x13715764355bdedb,
            ]))],
            [Felt::new(BigInteger256([
                0x939a609913e55969,
                0x558d99d2f40de511,
                0x62da0558bbdaa832,
                0x34289add7df049d1,
            ]))],
            [Felt::new(BigInteger256([
                0xf1423de9d9a05ddf,
                0x54f5f8c29b111130,
                0x318dc5fea0e19b20,
                0x28eb318437610fd2,
            ]))],
            [Felt::new(BigInteger256([
                0xf67c3cf6e026aa11,
                0xf4363e73909241c6,
                0xa256957740719169,
                0x20b92231b43912c7,
            ]))],
            [Felt::new(BigInteger256([
                0x1cff8cc69f222124,
                0x9990b45a2533cafd,
                0xfe181e97d9661625,
                0x055748904227f744,
            ]))],
            [Felt::new(BigInteger256([
                0x5b44a194f484931d,
                0x2c1ddb5cc7bf07ee,
                0x28b3276ac00d2f12,
                0x057435c8f09835c5,
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
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x94e74e0f54b6b887,
                    0xd017ca3a17070d98,
                    0xacfa10793830d976,
                    0x0e7cf4e25aaa4a15,
                ])),
                Felt::new(BigInteger256([
                    0x1d05f7fd42ed3126,
                    0x3d1fd5687ebbb755,
                    0x607529534e0b5a5b,
                    0x25c7b09989ac518a,
                ])),
                Felt::new(BigInteger256([
                    0x3eaf7e50d42f3c33,
                    0x3f553ed4b95c5301,
                    0x39cfaa803d34644d,
                    0x0e8c49e2aa33b819,
                ])),
                Felt::new(BigInteger256([
                    0x7043955401c88908,
                    0x6f8854d6dbc6e7e8,
                    0x1f2e18bd43e1240c,
                    0x080f6ae5d3836af9,
                ])),
                Felt::new(BigInteger256([
                    0x8f0e58faf0d38e98,
                    0x3e7e727dfec71960,
                    0xfe6b6cb3e0804693,
                    0x0fc7610e6e953a03,
                ])),
                Felt::new(BigInteger256([
                    0x3099c70671857d1d,
                    0x2ca823ab2471c72c,
                    0x14efa23151ea4e33,
                    0x06c90d783eea4994,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf91270076b78341b,
                    0x21ac0ce7b33bf3b7,
                    0x3ad5caec89a1544b,
                    0x36594de36250275f,
                ])),
                Felt::new(BigInteger256([
                    0x0589e410f71e1f78,
                    0x2360b0013a541314,
                    0xfb752cded30d8330,
                    0x027733e75fe2c070,
                ])),
                Felt::new(BigInteger256([
                    0x74f50f7c4ac352b8,
                    0xd01832f4aff30de7,
                    0xcb39a36d8e532f0c,
                    0x130317dde4f201a0,
                ])),
                Felt::new(BigInteger256([
                    0x7eb702c2997cd8ec,
                    0xe2ee4e5043300fbd,
                    0xc882a043da7e4eeb,
                    0x13387653fd740a8d,
                ])),
                Felt::new(BigInteger256([
                    0x5eab938ab97fbd9c,
                    0xd339b05537903a49,
                    0x333bee64aca6c7c4,
                    0x2f9fb6442d504331,
                ])),
                Felt::new(BigInteger256([
                    0x4d01037e04be8204,
                    0xc7df971546f88e7c,
                    0xf354516e9d37dc95,
                    0x1f3704a07ab404ec,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x34dabf585d878c5c,
                    0x22f5b4a58e0c7130,
                    0x8b0b8ee21a076269,
                    0x199889764c3d9779,
                ])),
                Felt::new(BigInteger256([
                    0x23eb30f0f8963833,
                    0xbdb7348b21c6e150,
                    0xdbe9a76cf206feaa,
                    0x2324a70dd34a085c,
                ])),
                Felt::new(BigInteger256([
                    0xa0cf488cba2af4ab,
                    0x1e463d4c6ee33502,
                    0xb631c42566f4f8b5,
                    0x1a9288b9bf1a16fe,
                ])),
                Felt::new(BigInteger256([
                    0xaf2c571f0ee77cb0,
                    0xc6e4f51ccbdbd84d,
                    0x440444001376ce04,
                    0x18483fc0816f5920,
                ])),
                Felt::new(BigInteger256([
                    0x59cf4f82e2582922,
                    0x9f28c6f186321871,
                    0x66748d41382712c9,
                    0x0db13269f42a5c3c,
                ])),
                Felt::new(BigInteger256([
                    0xebb4d330026e1a50,
                    0x5daf455184d75c5d,
                    0x1bc55532e5305e7c,
                    0x1d3ed807f427daa6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x500c126b61b7b0b6,
                    0xb8ab46ef2fecca5b,
                    0xc3ae4c5d25605e34,
                    0x1bf44740d1b92cc2,
                ])),
                Felt::new(BigInteger256([
                    0x5f1025f2601a2f9f,
                    0x11c225aa6395c4cd,
                    0x8ec15b9ecf3c0fb1,
                    0x0b71f6ffcc75ed1c,
                ])),
                Felt::new(BigInteger256([
                    0x23afa790827ed478,
                    0x8a5acdf9d15b4755,
                    0x677527734f2026b3,
                    0x328dcde7246cf81c,
                ])),
                Felt::new(BigInteger256([
                    0x4c470c93d483ab87,
                    0xc05d60227abf401a,
                    0x8f087a6683387860,
                    0x2bb1c1f522aacc35,
                ])),
                Felt::new(BigInteger256([
                    0xb56a507f25a7e522,
                    0x5a370e37a0f61a46,
                    0xb49c028ecffc904d,
                    0x3898429b2ce33cce,
                ])),
                Felt::new(BigInteger256([
                    0x52b1dcf09a697be5,
                    0xbd5e15f1d25667f5,
                    0x76ccd52991b3dc5f,
                    0x19507bc5b9222bc5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x46dc73d7e062d5d0,
                    0x82e97f33870a2171,
                    0xd1a23519f275ec31,
                    0x29662b3b54d58f44,
                ])),
                Felt::new(BigInteger256([
                    0x447acdd08f3e5523,
                    0xac7291f7d9229959,
                    0xfc4dcd4998e96bea,
                    0x018f1a46b8774885,
                ])),
                Felt::new(BigInteger256([
                    0x8d18e4b47614f8dc,
                    0xed6717032aeca449,
                    0x3ae3aa4d13869b13,
                    0x1f224106150081f4,
                ])),
                Felt::new(BigInteger256([
                    0x140141253606a62d,
                    0x6a7725e7377f1e39,
                    0x9b7fe4937dfc8f86,
                    0x00aff6bdbff36056,
                ])),
                Felt::new(BigInteger256([
                    0x435f2f3025d52435,
                    0xdfb106460795b7b3,
                    0x2cf09dcff16886e0,
                    0x0a6b3986aa2346f9,
                ])),
                Felt::new(BigInteger256([
                    0xa9e2fa02588dd4dc,
                    0x36e45be7b8e37923,
                    0x806af9f555355d46,
                    0x1275a86d6584f7a7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x75e245b6d954c574,
                    0x3acde1506feadb37,
                    0xe2f9eb35c2361984,
                    0x220c25da0f1807d8,
                ])),
                Felt::new(BigInteger256([
                    0x098902724a954cf5,
                    0x67669e4ab9bef2d2,
                    0x7e73233f24fe5642,
                    0x33586890cc4d239f,
                ])),
                Felt::new(BigInteger256([
                    0xd94f9246e7b81784,
                    0xa1c891de53620932,
                    0x99492390a4e06ee6,
                    0x20864121d1e0c106,
                ])),
                Felt::new(BigInteger256([
                    0x23a0584aac1e65dc,
                    0xb96404fefcbac480,
                    0x0c5477db7a600c20,
                    0x1cbf00683a4200e3,
                ])),
                Felt::new(BigInteger256([
                    0x33639754352ccb65,
                    0x7a418c0aaafa6611,
                    0x5ae24b9c3e7dd330,
                    0x1b3cf5c5ebafd05b,
                ])),
                Felt::new(BigInteger256([
                    0x11c7c09152d6c591,
                    0xa2fd2b661308e282,
                    0x675df2dae26d26c3,
                    0x04c3986fbde7700c,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x127a68ec2ef0786e,
                    0x9b283cfdbd967e52,
                    0xcbe22f617cdf2ac9,
                    0x2debc7ddc64dc00a,
                ])),
                Felt::new(BigInteger256([
                    0xc7202bb5114e01fb,
                    0x840141d2f688007d,
                    0x65bfe1217ca126f7,
                    0x004a6e1a10ff8267,
                ])),
                Felt::new(BigInteger256([
                    0xd22e028568376390,
                    0x430ed52996566ac5,
                    0xc4944bda389d5e1b,
                    0x16fc6c7697ea9af2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd0f8c88cb5da2f5d,
                    0x5e24a94e61bf20b8,
                    0x92b99d47e3fd1f3b,
                    0x253afbe875ac0162,
                ])),
                Felt::new(BigInteger256([
                    0x95fbe2261f782e02,
                    0x208f8461fe9d907d,
                    0x7cccaba3d3f60fcf,
                    0x21a058146dd86822,
                ])),
                Felt::new(BigInteger256([
                    0x72764fd84aa1bf15,
                    0xaae4b6829ca32a65,
                    0xa95d3943bfa12de4,
                    0x253ba659425ab0a1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x174aa7ebc8d42ae1,
                    0x2435a00ba1a0e007,
                    0x16398a43c19b488d,
                    0x07d54e949d553974,
                ])),
                Felt::new(BigInteger256([
                    0x4f6adbedb9eacae7,
                    0x1e262e869b0a159e,
                    0xe0447f75bab48b4e,
                    0x26b5a8f5d97f0a66,
                ])),
                Felt::new(BigInteger256([
                    0x2a7943b25abe6167,
                    0xf7ad03eeb004e569,
                    0xc16327280603bbd0,
                    0x15abf7858a9874a9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x693072088128ec70,
                    0x74c473dad41fb7aa,
                    0xded64f640b814b85,
                    0x304e77d884178e7e,
                ])),
                Felt::new(BigInteger256([
                    0xab6f8d463d60e3a2,
                    0x8280a0a615fde7ca,
                    0x6d69392aafaa8417,
                    0x30b2a0466388b8a3,
                ])),
                Felt::new(BigInteger256([
                    0x37238bf3234b0921,
                    0x24bbde21a3eae71f,
                    0xa289415d5cecf007,
                    0x2d87bdc3ae892573,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x16e7f314341820e0,
                    0x92ad172c3ee58d40,
                    0xfc6666ea03d90c80,
                    0x286fd697c4704ee4,
                ])),
                Felt::new(BigInteger256([
                    0x574a60302e7858f7,
                    0x6ae9292361139d3e,
                    0x699ce463a4315f45,
                    0x0abe649f20417a2a,
                ])),
                Felt::new(BigInteger256([
                    0x98f19f31312ced98,
                    0x7d3732ffaa57d51a,
                    0x22611e4383b2afc9,
                    0x32c543d8799bfbea,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x35db806f6d4de551,
                    0x487aa29b5bd69bb0,
                    0xc489c35484ba337b,
                    0x01592d37890b468b,
                ])),
                Felt::new(BigInteger256([
                    0x60f2bda2f77da3c7,
                    0x8565a953e0c49acf,
                    0xbdce1ccdf30adcf8,
                    0x355fb1f759c5bb32,
                ])),
                Felt::new(BigInteger256([
                    0x917179c0ee6dd877,
                    0x1597c75e76206b1b,
                    0x01bfa8bfe41e87d8,
                    0x116ef942e9f25c09,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7a10f5393642954e,
                    0xae5d2e51364a2b1c,
                    0x80a336bd531f6447,
                    0x0cc45c089550200e,
                ])),
                Felt::new(BigInteger256([
                    0x8460089cd9850717,
                    0x33cbaad0862ec17d,
                    0x73cebf2df5c2069a,
                    0x38af4fa957039ef0,
                ])),
                Felt::new(BigInteger256([
                    0xb4d1e07fc751f3cf,
                    0xbc9d86ab3baa6754,
                    0x02990bd7ac3e3aa8,
                    0x18dca1607be6c716,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7dc23e622ce32b4b,
                    0xcc9b22cca3187a1d,
                    0x0a469a0468a6ab46,
                    0x2cb827db25b2a99b,
                ])),
                Felt::new(BigInteger256([
                    0x25dcc69495687966,
                    0xcb2ce32477892a70,
                    0x2b89055f4ec34e84,
                    0x2783f674d83311de,
                ])),
                Felt::new(BigInteger256([
                    0xfd4c2abb93ddf4e8,
                    0x4f7aaa6d5ce3e035,
                    0xbe36db7646ffc81b,
                    0x237798983d746a7f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5e14022cda697335,
                    0x29f9096cafe3bc15,
                    0x8c88df7335fbc3e4,
                    0x310ff7bf7f5ab5f4,
                ])),
                Felt::new(BigInteger256([
                    0xfff743f21afaa3b8,
                    0xf8a33dd421e890bc,
                    0x8310fe53637dd096,
                    0x0dfb4b66305728fb,
                ])),
                Felt::new(BigInteger256([
                    0x2b1663f7cb4b4b03,
                    0x9058a5ce95a0a2aa,
                    0x9321cff4053ff9d9,
                    0x29485c00e34ede25,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x33d39e1852c70b8a,
                    0xec4e236b008d7ce0,
                    0xcc981f3834c74878,
                    0x0c271b482fb0537b,
                ])),
                Felt::new(BigInteger256([
                    0x0640a8b16c6a8261,
                    0x238e195fba7dca2b,
                    0x730b409eb0927cb4,
                    0x2d37eb6a34f846bd,
                ])),
                Felt::new(BigInteger256([
                    0x2a666274693375a3,
                    0x3c27a604e180367d,
                    0xf10499db5234a463,
                    0x39c0ff5fbe6ea99a,
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
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6d3c8371d4090bca,
                    0x55d32c69242eae0b,
                    0x0254495d222fc254,
                    0x1f97f835fe1e2f3a,
                ])),
                Felt::new(BigInteger256([
                    0xe3829a8c798f223b,
                    0xbd6212a8ecec4198,
                    0x61bc124707dd5c83,
                    0x0300ff9140083071,
                ])),
                Felt::new(BigInteger256([
                    0x95f435bb175fd8f7,
                    0xa217806382c407b1,
                    0x97d63c8ea66f69df,
                    0x13bd11658a3b7543,
                ])),
                Felt::new(BigInteger256([
                    0xf0407f42fb87ffb7,
                    0xa0afb7f53ae3c471,
                    0x64576e29173c7649,
                    0x05e860f22cea3639,
                ])),
                Felt::new(BigInteger256([
                    0x32113b8263352c16,
                    0x417de7079737df63,
                    0xe6afb8326828c0a8,
                    0x2b4de1494501a576,
                ])),
                Felt::new(BigInteger256([
                    0x2adc009eb3539ccc,
                    0xff58f2a8de12308a,
                    0x90ed8759b3a74ec1,
                    0x32059aeef463b2ab,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x869bc61b6193fc56,
                    0x5acebec02a8b94ce,
                    0xb9cb981d474a22c5,
                    0x0db8417ca7526eed,
                ])),
                Felt::new(BigInteger256([
                    0x0f44a4f4dde646f6,
                    0xaffa021d677c80a5,
                    0x9309d6f0f12649a2,
                    0x1edeeb4292d10381,
                ])),
                Felt::new(BigInteger256([
                    0x35fac5e9d3e963a4,
                    0x03b9e5af1f4c9e60,
                    0xfdccfc9771175524,
                    0x30e2c82169258bee,
                ])),
                Felt::new(BigInteger256([
                    0x2f8ca0a3201e5e58,
                    0x326ef93f5fe753a4,
                    0xe1e40c8937ca58cc,
                    0x2ae951fa6cf94d64,
                ])),
                Felt::new(BigInteger256([
                    0x49e31857a012834d,
                    0x08267832aafe56c2,
                    0x2ba5b07b61abac0e,
                    0x25bc56942329b727,
                ])),
                Felt::new(BigInteger256([
                    0x5d13b4e331b11ceb,
                    0x19cfa8ed92f027a8,
                    0xb150ce2d7b9aa8fa,
                    0x3941352ab1a72de3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x96a84870083cd487,
                    0x4caba0320f1b8b31,
                    0xd3e68f1e047148fe,
                    0x3d2dc4b1d2dc257b,
                ])),
                Felt::new(BigInteger256([
                    0x6f64682743ff12db,
                    0x0bd51999f9775e58,
                    0x5fe62c92371a922a,
                    0x08831e36d167386c,
                ])),
                Felt::new(BigInteger256([
                    0x9ea125351db4bab0,
                    0x9975fdf2a460b0be,
                    0x4e434ac1374404a7,
                    0x2ecc05a235449588,
                ])),
                Felt::new(BigInteger256([
                    0xb330d61f53334725,
                    0xe88fa14f77624d00,
                    0xd329fe0553ec2028,
                    0x3e789a36e6fce3d2,
                ])),
                Felt::new(BigInteger256([
                    0xb1eab381a68cf328,
                    0xb1472c333a74982f,
                    0x26c8b724dc2decb0,
                    0x073d937f33feac3b,
                ])),
                Felt::new(BigInteger256([
                    0x4c396c6231ff4537,
                    0xd490aa1139d56850,
                    0x258ae6477e60e755,
                    0x2ff97092530f45e4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3919c24b5d95cf88,
                    0xa08630993d6ac649,
                    0x0ff19b717ee9843b,
                    0x14dae93c60564f10,
                ])),
                Felt::new(BigInteger256([
                    0x9d87cf95d3741f58,
                    0x889967bfb232b04a,
                    0x35e5d75db6c83935,
                    0x2185c18db8214b80,
                ])),
                Felt::new(BigInteger256([
                    0x7c272841d6eb419a,
                    0xf5f45d38c13e42f6,
                    0xb3defdfd478619d4,
                    0x25629d257d9baa88,
                ])),
                Felt::new(BigInteger256([
                    0xb367f4e22122192e,
                    0x5c5c893c69739579,
                    0xcd99d643704a4c3e,
                    0x1431c850177bdc11,
                ])),
                Felt::new(BigInteger256([
                    0x65b883d8374afd6e,
                    0xa5b0651804014dd3,
                    0xc04cecc0e8a92976,
                    0x27a3df868a7f2487,
                ])),
                Felt::new(BigInteger256([
                    0x5df48d2d99fbabe3,
                    0x861e5b8a205e5073,
                    0x8e127023b682a8b4,
                    0x0bc2114bf7ef7b2b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb384c36f46f689b0,
                    0xaa29e948010b3ba7,
                    0x26f2798def811536,
                    0x3b2b1eef21dc0cc4,
                ])),
                Felt::new(BigInteger256([
                    0x01a2300b13fb1e53,
                    0x8c0909200647d818,
                    0x34b262bebaeb0972,
                    0x0aa0ec7046d1f11c,
                ])),
                Felt::new(BigInteger256([
                    0x4da189292bb6d4ed,
                    0x86c7dd3a8e13f94a,
                    0xd1cee538ed393ad0,
                    0x036ae8e4f9a29169,
                ])),
                Felt::new(BigInteger256([
                    0x0127aa27c81521dd,
                    0xbe44710466237d2e,
                    0x3a2c712f1f8ef280,
                    0x2a656bec91a5d3fd,
                ])),
                Felt::new(BigInteger256([
                    0x9d6329ff5f09240c,
                    0x46a4a82c3ffde816,
                    0xda0c501042e79ddb,
                    0x044ce62fcdbd3ad1,
                ])),
                Felt::new(BigInteger256([
                    0x2a959d3641bb5504,
                    0x87d249e72b78ccbb,
                    0x614ac7366d750994,
                    0x1823a499b52a27b7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x04fcc374f85e5138,
                    0x63ff71d53ca22fcb,
                    0x1a213bb31e706823,
                    0x09282fd9afb66889,
                ])),
                Felt::new(BigInteger256([
                    0x4bfab4065c586e45,
                    0x1acf4ab730fe8bfd,
                    0x48a8aaa4a4dca1aa,
                    0x3c0e2c7c6c85b964,
                ])),
                Felt::new(BigInteger256([
                    0xcef05f85c8e3086d,
                    0x18a652ecf1b9628c,
                    0x9bb4ae3bbbf12c80,
                    0x2c39c22302142b34,
                ])),
                Felt::new(BigInteger256([
                    0x351bd4f57c77728e,
                    0x0b02ac9db1a1e1e0,
                    0x94f75fb94f2e6edf,
                    0x3814e8c219146d44,
                ])),
                Felt::new(BigInteger256([
                    0x3cd053e615dcf36c,
                    0x2baf919414d25ca1,
                    0x03928e7e3580be35,
                    0x05d637bde99db822,
                ])),
                Felt::new(BigInteger256([
                    0xbd71e1cb777cabac,
                    0x0c20ed9db70b88ad,
                    0x93eb8bc079e62742,
                    0x06a949ed9afc9850,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x1f81ac05a875ddf8,
                0x3ff1bafe40e040b8,
                0xf6365c5d321dafdc,
                0x0532a26e6f37dd64,
            ]))],
            [Felt::new(BigInteger256([
                0x4d240f6a1ff41c73,
                0x07524b36f36b32be,
                0xb8e3822f77945cef,
                0x2c16fa5625df1a26,
            ]))],
            [Felt::new(BigInteger256([
                0x04e7dc6add7d572e,
                0x17c23984e31b3231,
                0xb7e130e182538fac,
                0x0436ef10016cb884,
            ]))],
            [Felt::new(BigInteger256([
                0x3335b4ffe1d4d931,
                0xd773c0aa7adf34d9,
                0xeec8c9ec1818bfa3,
                0x0e88d5e296296c95,
            ]))],
            [Felt::new(BigInteger256([
                0x4bd9c0e3e5823b24,
                0xa10666593328864c,
                0x36f2e1be19003e8d,
                0x3e7ff515b83029f6,
            ]))],
            [Felt::new(BigInteger256([
                0x846dafd4006b95f9,
                0x606390a1895f7ece,
                0xc7f9f1d38ca5f88d,
                0x1713a2b648f5045d,
            ]))],
            [Felt::new(BigInteger256([
                0x4fd0b6d5bfe596a7,
                0xc3700eaa1c5aacb9,
                0x49114b55054a0e6c,
                0x3c88c49fc163e1bd,
            ]))],
            [Felt::new(BigInteger256([
                0xf21b0cecae1a0ae7,
                0xc52ddabadcc9f9ef,
                0x3c893118039c65cc,
                0x181b75e4c75875e0,
            ]))],
            [Felt::new(BigInteger256([
                0x058a295d2074f9bd,
                0xab9dc10f049daf90,
                0xeaa65d9b976651fc,
                0x2ca75176323baa02,
            ]))],
            [Felt::new(BigInteger256([
                0x2eb9f4188b030709,
                0x6ea6397616a110a0,
                0x159c45486c681949,
                0x2311867880f774e0,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
