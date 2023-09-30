//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiPallas_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiPallas_4_3 {
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
            state[i] += Felt::from_le_bytes_mod_order(&buf[..]);
            i += 1;
            if i % RATE_WIDTH == 0 {
                AnemoiPallas_4_3::permutation(&mut state);
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
            AnemoiPallas_4_3::permutation(&mut state);
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
                AnemoiPallas_4_3::permutation(&mut state);
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
            AnemoiPallas_4_3::permutation(&mut state);
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
        AnemoiPallas_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiPallas_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiPallas_4_3::permutation(&mut state);

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

        let mut state = elems.to_vec();
        AnemoiPallas_4_3::permutation(&mut state);

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

    use super::super::MontFp;
    use super::*;
    use ark_ff::BigInteger;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![MontFp!(
                "21639214970066534150430088710672344339351825377325572601134230847435855090018"
            )],
            vec![
                MontFp!(
                    "14308310011320271075241074852554551793116589797873291197377178969597814757745"
                ),
                MontFp!(
                    "22812519244635639534870240978594413497217364395500311893561032045384117799878"
                ),
            ],
            vec![
                MontFp!(
                    "23765623204622958643368508730446153544497463659497977322294049803725089680077"
                ),
                MontFp!(
                    "2761657738390976099784033013656492009234327105692076199686725552553595160512"
                ),
                MontFp!(
                    "12354800733798554344880652102324243621499865971329973211194509884277810483989"
                ),
            ],
            vec![
                MontFp!(
                    "10033669896525688739559423792041083073657230262399786332112917151491349621156"
                ),
                MontFp!(
                    "2402935293599703694925481097313849400153027911079819709951633229930463274411"
                ),
                MontFp!(
                    "15141140496634667073241089391482623820656817856452117757862613027185176151320"
                ),
                MontFp!(
                    "9230081170729671270782214301227543693465268732972655430727126469580001066836"
                ),
            ],
            vec![
                MontFp!(
                    "16580475169231560943994826549979368689782367272071060833785390675083215453194"
                ),
                MontFp!(
                    "10004412817097373061517532903858807858993405489168929589455000884154966759320"
                ),
                MontFp!(
                    "1355194641053070000075197911936524025071372126913817851573375822588356336676"
                ),
                MontFp!(
                    "17444611727166347460082061146955620567496772682030127308229081870716301202092"
                ),
                MontFp!(
                    "15440790176274981693833551806629298448680841788514608132286069353289615622320"
                ),
            ],
            vec![
                MontFp!(
                    "15409207566483766714933995237189997770256127690043059752865384785081338806008"
                ),
                MontFp!(
                    "5751195299622947836280217965819888862825707754086574762028976072155459214669"
                ),
                MontFp!(
                    "804826448337405805732843527912622948408773912073805278584653490742562799174"
                ),
                MontFp!(
                    "54411331332446143311253369710064725692225955708390037299651399950515126386"
                ),
                MontFp!(
                    "22417573194360036812332079051361603326904698011705537374783188747808796361206"
                ),
                MontFp!(
                    "6469054496641404924286933338508030164119881143197330664753674015245329522845"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "18667893566940377934413531954661629005239155214336217717980971841120036720992"
            )],
            [MontFp!(
                "16332064307906021858995989073185923369241423502144838929406031304186177756421"
            )],
            [MontFp!(
                "10120813506782340547268980585023925831088423627255340408061228272104188937028"
            )],
            [MontFp!(
                "12573228387563430057694757326408676853111531942075183709320724792995739113307"
            )],
            [MontFp!(
                "21694522994823424510392828412675056114249469142972397236297530118523556182029"
            )],
            [MontFp!(
                "6766589173654963124934839474048684982288045583229496204050981767644297021519"
            )],
            [MontFp!(
                "1842089326709875619803844739111822363096300193007869419513139366580669412870"
            )],
            [MontFp!(
                "5302005038874182146455963492580261694651482438322761024988716317705130000976"
            )],
            [MontFp!(
                "17068290522837623179589332515070046988254553107483674469540861992224659316863"
            )],
            [MontFp!(
                "28627431873786754713714284091089204081618742879828933488615932584482648991866"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiPallas_4_3::hash_field(input).to_elements());
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
            [MontFp!(
                "18667893566940377934413531954661629005239155214336217717980971841120036720992"
            )],
            [MontFp!(
                "16332064307906021858995989073185923369241423502144838929406031304186177756421"
            )],
            [MontFp!(
                "10120813506782340547268980585023925831088423627255340408061228272104188937028"
            )],
            [MontFp!(
                "12573228387563430057694757326408676853111531942075183709320724792995739113307"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 124];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);
            bytes[62..93].copy_from_slice(&input[2].into_bigint().to_bytes_le()[0..31]);
            bytes[93..124].copy_from_slice(&input[3].into_bigint().to_bytes_le()[0..31]);

            assert_eq!(expected, AnemoiPallas_4_3::hash(&bytes).to_elements());
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
                MontFp!(
                    "155260004806059969980061416423330881889240336623781546121410760287809104774"
                ),
                MontFp!(
                    "18911999830037716111748566585472628944083385984045482994538443407116433259481"
                ),
            ],
            [
                MontFp!(
                    "10574950278057377054483945886502479610095784600476876759205777595522024326848"
                ),
                MontFp!(
                    "878944299778645826471008564793909673728274386951687124423989547665441563258"
                ),
            ],
            [
                MontFp!(
                    "19987739087819865782694564672591379265397013990789950980715436474624395508317"
                ),
                MontFp!(
                    "10174893348081987242771600544387711673877577306100134825491790172317730671617"
                ),
            ],
            [
                MontFp!(
                    "21195260622338728819653286536837364927496937117816249105425751540779649358326"
                ),
                MontFp!(
                    "17916063567746091983605399379505344079993739565625964333895436713994194878281"
                ),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "19067259834843776081728628001895959825972626320669264540659854167404242364255"
            )],
            [MontFp!(
                "11453894577836022880954954451296389283824058987428563883629767143187465890106"
            )],
            [MontFp!(
                "1214610126572804169573418964807113975911534814948525090252549882592158549597"
            )],
            [MontFp!(
                "10163301880755771947365939664170732044127620201500652723366511490423876606270"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_4_3::compress_k(input, 4));
        }
    }
}
