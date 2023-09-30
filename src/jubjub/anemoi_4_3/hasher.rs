//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::jubjub::anemoi_4_3::AnemoiJubjub_4_3;
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiJubjub_4_3 {
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
                AnemoiJubjub_4_3::permutation(&mut state);
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
            AnemoiJubjub_4_3::permutation(&mut state);
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
                AnemoiJubjub_4_3::permutation(&mut state);
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
            AnemoiJubjub_4_3::permutation(&mut state);
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
        AnemoiJubjub_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiJubjub_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiJubjub_4_3::permutation(&mut state);

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
        AnemoiJubjub_4_3::permutation(&mut state);

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
                "19736911815663382719237239115786888395575970607072582643439733400861889127809"
            )],
            vec![
                MontFp!(
                    "31946759962175780374941261532958277798312448517961548858630877527765960894665"
                ),
                MontFp!(
                    "31680178724577489535529906488875063813014417633893660210126505779894153150302"
                ),
            ],
            vec![
                MontFp!(
                    "15100074709805963566161313617893289450863920634620118918341174303450723706505"
                ),
                MontFp!(
                    "26991075841797048094523911515386779089616660122394275435269652583647141162643"
                ),
                MontFp!(
                    "23761055101505274891261696792444779377068885548604844277573549011731223771684"
                ),
            ],
            vec![
                MontFp!(
                    "32105594753131030823905475653352743335424796763733160951166436046612010717055"
                ),
                MontFp!(
                    "24796431536350598774278045145436006634601298982968352798234443773996100888040"
                ),
                MontFp!(
                    "42548285289754577520238386221339202252725381906857009731131609145529353643750"
                ),
                MontFp!(
                    "51832059921719748524500148445537616852277071433800155947397646863193952240863"
                ),
            ],
            vec![
                MontFp!(
                    "1685287177962010948634847020397702753124281625366799951519105194053598124989"
                ),
                MontFp!(
                    "20174147333400258966356845230019322112808487184590891010553884228507683189189"
                ),
                MontFp!(
                    "42579515684805859653403026523244442411111597245467147557668580096906060147633"
                ),
                MontFp!(
                    "18464889611763124528088774889057062981994008860698292684284800102206531642091"
                ),
                MontFp!(
                    "27939464771368012549844909928951309357193379363471123307137671121916994869588"
                ),
            ],
            vec![
                MontFp!(
                    "26288506374337233237831346805552604438156800874545969417895226312740793936092"
                ),
                MontFp!(
                    "52134314164669899271836981617482339351659104456898310695906629048366557554107"
                ),
                MontFp!(
                    "11952069606878620857447002269771042274909344717517619770517608485610969643597"
                ),
                MontFp!(
                    "12708257021021527694474435639668406076795082041062708643824131522456656513197"
                ),
                MontFp!(
                    "35510257625349750875854515723154755203101510280155377118503587497338324056732"
                ),
                MontFp!(
                    "22119213328391132574776266432746305056001872591008623325677435512301778001010"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "38310213048643262329622760360212489938704092112371866787594917193813139289809"
            )],
            [MontFp!(
                "45147274098615966749487358174073372009888068879405542868378683891398912045625"
            )],
            [MontFp!(
                "4785849960602933515566711981483873501178725314523130166590869761297475734119"
            )],
            [MontFp!(
                "1486774069014101020044227815788714221463930175285233978152612010420134052123"
            )],
            [MontFp!(
                "4172618539027921782691550244087389997928769737233915647018981169427679869727"
            )],
            [MontFp!(
                "49986636652664727166440151191423871759052762592624834821825260055659041109954"
            )],
            [MontFp!(
                "40142577054204156778386279699310575662455349467901388696701183353591683853662"
            )],
            [MontFp!(
                "40870542921816677783831683071896086844665133372200530165396327582519465382607"
            )],
            [MontFp!(
                "3972329888537329495228388652202562079959194676217386210995554592077488684413"
            )],
            [MontFp!(
                "33786031787078512059780857959220183841291296052236835108016737551805429129482"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiJubjub_4_3::hash_field(input).to_elements());
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
                "38310213048643262329622760360212489938704092112371866787594917193813139289809"
            )],
            [MontFp!(
                "45147274098615966749487358174073372009888068879405542868378683891398912045625"
            )],
            [MontFp!(
                "4785849960602933515566711981483873501178725314523130166590869761297475734119"
            )],
            [MontFp!(
                "1486774069014101020044227815788714221463930175285233978152612010420134052123"
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

            assert_eq!(expected, AnemoiJubjub_4_3::hash(&bytes).to_elements());
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
                    "48136361849153243738173322980436308194640949106271265019940984579279517354580"
                ),
                MontFp!(
                    "49817329994699533652293260449422143653559101821826781651581903522052933662147"
                ),
            ],
            [
                MontFp!(
                    "18205127879525878941786319451857577932430192362546047236582468460825047231827"
                ),
                MontFp!(
                    "21453885597127632191822341381145144602006860020065635625097486765368206615492"
                ),
            ],
            [
                MontFp!(
                    "48653614180029901857167901490666718526822512985869211183802509495387476230804"
                ),
                MontFp!(
                    "13120598758059651227508017984124942960292547395955043822203068042983899773744"
                ),
            ],
            [
                MontFp!(
                    "16106234441029043570369654455645100385516059460850491304710352948623471075628"
                ),
                MontFp!(
                    "12480629909892753583280577597892300648448114904435781136420871444940482192828"
                ),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiJubjub_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiJubjub_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "45517816668726586911018842921672486010509498427570408848919229401393869832214"
            )],
            [MontFp!(
                "39659013476653511133608660833002722534437052382611682861679955226193253847319"
            )],
            [MontFp!(
                "9338337762963362605228178966605695649424507881296617183401918838432794820035"
            )],
            [MontFp!(
                "28586864350921797153650232053537401033964174365286272441131224393563953268456"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiJubjub_4_3::compress_k(input, 4));
        }
    }
}
