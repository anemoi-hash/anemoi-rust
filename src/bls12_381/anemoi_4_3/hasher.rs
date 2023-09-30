//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBls12_381_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiBls12_381_4_3 {
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
            state[i] += Felt::from_le_bytes_mod_order(&buf[..]);
            i += 1;
            if i % RATE_WIDTH == 0 {
                AnemoiBls12_381_4_3::permutation(&mut state);
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
            AnemoiBls12_381_4_3::permutation(&mut state);
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
                AnemoiBls12_381_4_3::permutation(&mut state);
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
            AnemoiBls12_381_4_3::permutation(&mut state);
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
        AnemoiBls12_381_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiBls12_381_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiBls12_381_4_3::permutation(&mut state);

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
        AnemoiBls12_381_4_3::permutation(&mut state);

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
            vec![
                MontFp!("3478190366645077329062387911759857236499263186749383695250484804369076098962408026507385944138785122527147895559085"
            )],
            vec![
                MontFp!(
                    "1204757022733618702245695022362210673090184180289461640964393670508762664913657763815285216675013204672855455968166"
                ),
                MontFp!("229264651799887865578273118032416158552374500069538974023134241671888891566565325016261970533772885892514131522324"
                ),
            ],
            vec![
                MontFp!(
                    "177717861685121075852882639397771234670672217624656643704253853166625554591784186707288011666580142027095461219927"
                ),
                MontFp!("2664159978722322727411461850719183495029060572269930286611223738648050026900916021701195336148073915673920578139515"
                ),
                MontFp!("3168897365696464751089641952623368003358434783685755752721979882688778293819456233336426355170181665478528538972725"
                ),
            ],
            vec![
                MontFp!(
                    "529423574900204087999767742321637746109193723419458908707428479135997266319072674410850341834328721890278957792618"
                ),
                MontFp!("3148159782614182473118320567803373093488104354203130300286617668600785071681974546798837662949302481963079067906680"
                ),
                MontFp!("1083645939082345614904425670523907746448517423057464053705787555890054715287719463530138543606015991452224492494731"
                ),
                MontFp!("2807268919105297081646910924341136067920274853031997221954508076679997895846027148765771991535533447647305434261916"
                ),
            ],
            vec![
                MontFp!(
                    "3942124505372145879707769758437029629242473037070041756829399827985529317490119201593294288730321858969168795882591"
                ),
                MontFp!("2948389218971913142595299953737210328251053567408899560831733646698737436722062772042039437496555401586890558844020"
                ),
                MontFp!("1951166782335181494088937508317393496993611622685086968055175772193804505322387296392690938471020669718383975309532"
                ),
                MontFp!("3159957546857530858381811628455870492082054267240750742256782466302933919683184838401730844715872139634403069726858"
                ),
                MontFp!("2862834010401599898219710787596316847120613900794954135737067214558874858756329396400855600235963759882747786540404"
                ),
            ],
            vec![
                MontFp!(
                    "442383683928193282947541811118171577253029396096980687266816333308650666919596182616073814966705645577455262083516"
                ),
                MontFp!("3474343543165004187936129993054631267742133607255083236479107754787485302127789221230133641816397320407994932786163"
                ),
                MontFp!("3380527423356682362307992583106461603599581627552321227216704820347353004760999228306110918204127470701240112195941"
                ),
                MontFp!("3112951976232687026802036550687130608286117306290466630678325309622306021784306332142592082742828936853603170941922"
                ),
                MontFp!("2611204297476868862193359640196948819406140961326550374402004429141754814188552985163011279977823062470634807546903"
                ),
                MontFp!("2889443312718009524459939344276095701899215302717389345365446369440025952961431535196040082429742136085575408972705"),],];

        let output_data = [
[
    MontFp!("2940668067266832030903300713880139151173275526618596596954076114821895049272625792093682693181482523953691902365663"),],
    [MontFp!("3342678617705459012287997353144186876092786389171849821793537400751167280639654962279805649671173411894902355858415"),],
    [MontFp!("897668485411151016252063836411911079925767629775867831728887685001424286000176738706190210972231148728580437604935"),],
    [MontFp!("261101949137956463543477394252410370931502266322987592818386276127634601896308935037921959975558135101361870359258"),],[MontFp!("2245100416982495102199387963939381924471363148632927465707066694849817699597209269935143147911791023854903195819366"),],[MontFp!("1992338069801663438012219224631570563826476479512040972197716217632689222758336302238924878425504523764403629814367"),],[MontFp!("877634473926936284841537119546245960890683255565853010165732801201337279756927477938178482170744533435377368968730"),],[MontFp!("3724599954884844853806409306911120960065460889492714025286924052647248345170780173516192322273128534317365723099745"),],[MontFp!("3200113069035622902050861397550093524076250368151123518340606353448676370384783100112330409231625375189488417196918"),],[MontFp!("2149147613891947510381404926601049758232543207193560917871022051839987633002652770226667468663320442620171354349106"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_381_4_3::hash_field(input).to_elements()
            );
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
            [
    MontFp!("2940668067266832030903300713880139151173275526618596596954076114821895049272625792093682693181482523953691902365663"),],
    [MontFp!("3342678617705459012287997353144186876092786389171849821793537400751167280639654962279805649671173411894902355858415"),],
    [MontFp!("897668485411151016252063836411911079925767629775867831728887685001424286000176738706190210972231148728580437604935"),],
    [MontFp!("261101949137956463543477394252410370931502266322987592818386276127634601896308935037921959975558135101361870359258"),],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 188];
            bytes[0..47].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..47]);
            bytes[47..94].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..47]);
            bytes[94..141].copy_from_slice(&input[2].into_bigint().to_bytes_le()[0..47]);
            bytes[141..188].copy_from_slice(&input[3].into_bigint().to_bytes_le()[0..47]);

            assert_eq!(expected, AnemoiBls12_381_4_3::hash(&bytes).to_elements());
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
[MontFp!("2182431290704436788418060634803748513006487864594071940133967216764562185869534818587870671419244594946258111655196"
                ),
                MontFp!("3584595477211783622630741012615941601587793370526228493195173031759026796058440775701817784781446148722285618960524"),],[MontFp!("3672701981707521318101679600213368747520920180417563081581162373949359365624895815123347476885913853820192970903029"
                ),
                MontFp!("2732651319344478048053404620785574688197427271909258305324791677897765083343328257136423854703039232453649848191428"),],[MontFp!("415713322322912545564193907405043626868156318882288129485197572128572554556771681528610943606168452623797653150692"
                ),
                MontFp!("2147536765737237170190209707679202535424442288937096923239685310680933052450050052160892705170564924404241540745120"),],[MontFp!("662760424293721908696422562222349787646979058578262140084405067201215405511285978285590137403678759847721259276360"
                ),
                MontFp!("1078728456415669970181374549496070873178175088025952973371266916547380812535592884145986873667653821102411901837124"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_381_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_381_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
[MontFp!("1764617212694553017631011821683785958037398415181292547997082112399557331437137729847000827071675079630649458055933"),],[MontFp!("2402943745830331972737294395263039279161464632387813501573895915723092798477386207817083702459937422235948546534670"),],[MontFp!("2563250088060149715754403615084246162292598607819385052724882882809505607006821733689503648776733377028039193895812"),],[MontFp!("1741488880709391878877797111718420660825154146604215113455671983748596218046878862431577011071332580950133161113484"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_381_4_3::compress_k(input, 4));
        }
    }
}
