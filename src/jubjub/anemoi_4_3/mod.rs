//! Implementation of the Anemoi permutation

use super::{sbox, Felt, MontFp};
use crate::{Anemoi, Jive, Sponge};
use ark_ff::{One, Zero};
/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 4 field elements or 128 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 4;
/// 3 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 3;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 2;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = 1;

/// The number of rounds is set to 14 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 14;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over Jubjub basefield with 2 columns and rate 3.
#[derive(Debug, Clone)]
pub struct AnemoiJubjub_4_3;

impl<'a> Anemoi<'a, Felt> for AnemoiJubjub_4_3 {
    const NUM_COLUMNS: usize = NUM_COLUMNS;
    const NUM_ROUNDS: usize = NUM_HASH_ROUNDS;

    const WIDTH: usize = STATE_WIDTH;
    const RATE: usize = RATE_WIDTH;
    const OUTPUT_SIZE: usize = DIGEST_SIZE;

    const ARK_C: &'a [Felt] = &round_constants::C;
    const ARK_D: &'a [Felt] = &round_constants::D;

    const GROUP_GENERATOR: u32 = sbox::BETA;

    const ALPHA: u32 = sbox::ALPHA;
    const INV_ALPHA: Felt = sbox::INV_ALPHA;
    const BETA: u32 = sbox::BETA;
    const DELTA: Felt = sbox::DELTA;

    fn exp_by_inv_alpha(x: Felt) -> Felt {
        sbox::exp_by_inv_alpha(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                MontFp!(
                    "32482331693491338912746715212804309955055762795010043428638444021114192055700"
                ),
                MontFp!(
                    "33492235896239677379318998127111915626105094764445420316563603803650677798024"
                ),
                MontFp!(
                    "40852276710209360508792143474575139079377611682724718279538612654089409265622"
                ),
                MontFp!(
                    "17454758553032001173448113737176508049912904137600528382743215109473773360326"
                ),
            ],
            [
                MontFp!(
                    "31697690150500698069891377539641413643042395767418268110622836737358937273733"
                ),
                MontFp!(
                    "14936182001098588566889068859187704904046509134953535684803012051039021870434"
                ),
                MontFp!(
                    "32062323223159678058199358518264370918121206315536336340273716710093585703133"
                ),
                MontFp!(
                    "43578186782945849558350591695416792423021334793450758127359220722483217294020"
                ),
            ],
            [
                MontFp!(
                    "2796790445465504069863714602407416422364873557800358379683833414928340335973"
                ),
                MontFp!(
                    "50511725679821395285809210187672021215784469444066791282435722882320350392529"
                ),
                MontFp!(
                    "50112751287890876405347006035294541681882836253933320043432504777600143388328"
                ),
                MontFp!(
                    "41948963963773661486224262314467774385457060500842151247193284107159670392404"
                ),
            ],
            [
                MontFp!(
                    "28260061041875208478247269981369481951055158365877458064135425915086050180900"
                ),
                MontFp!(
                    "7520225685750815501541377676439117981829586983869671058576304486742543973903"
                ),
                MontFp!(
                    "8494952710499081348150272321526897482079052146658268355582624763000698357905"
                ),
                MontFp!(
                    "18772239904024279219312393340394095284703500765493091002652166849996241645214"
                ),
            ],
            [
                MontFp!(
                    "25465323965862504985542346374283770884484816724541277131322984567785823028157"
                ),
                MontFp!(
                    "50757513099879106518371328917154053295290464515758867435875604163645682862350"
                ),
                MontFp!(
                    "14539035946163836242004436655669921153164985614622459585532484049600703380973"
                ),
                MontFp!(
                    "39702178203045326512186708524595197587076032375200691801935943638307284457274"
                ),
            ],
            [
                MontFp!(
                    "22325702346208357281572047271348765192338134444085978084526703993427739008304"
                ),
                MontFp!(
                    "24578203192850644218962499856076936668656423409248399287800807169536587510513"
                ),
                MontFp!(
                    "9321790521148453575621974423755572637302841662049080540134236232094996035474"
                ),
                MontFp!(
                    "14510393621755087595232205218034236411877322034688824965880542086718055104181"
                ),
            ],
        ];

        let output = [
            [
                MontFp!(
                    "14981678621464625851270783002338847382197300714436467949315331057125308909861"
                ),
                MontFp!(
                    "14981678621464625851270783002338847382197300714436467949315331057125308909861"
                ),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                MontFp!(
                    "26465412774926089111718321100574595527540964126946526979500696246159200663557"
                ),
                MontFp!(
                    "26465412774926089111718321100574595527540964126946526979500696246159200663557"
                ),
                MontFp!(
                    "39248768654128688292658764022179785046913739642751806076985032101719802651189"
                ),
                MontFp!(
                    "39248768654128688292658764022179785046913739642751806076985032101719802651189"
                ),
            ],
            [
                MontFp!(
                    "12740277426642017153170591585468114140384764416905683044706614677794597857620"
                ),
                MontFp!(
                    "12740277426642017153170591585468114140384764416905683044706614677794597857620"
                ),
                MontFp!(
                    "14891594025153715433052427327081456214862500696375893038292967289916437753110"
                ),
                MontFp!(
                    "14891594025153715433052427327081456214862500696375893038292967289916437753110"
                ),
            ],
            [
                MontFp!(
                    "14981678621464625851270783002338847382197300714436467949315331057125308909869"
                ),
                MontFp!(
                    "14981678621464625851270783002338847382197300714436467949315331057125308909869"
                ),
                MontFp!(
                    "52435875175126190479447740508185965837690552500527637822603658699938581184512"
                ),
                MontFp!(
                    "52435875175126190479447740508185965837690552500527637822603658699938581184512"
                ),
            ],
            [
                MontFp!(
                    "38749778788788807438157413537102975071476628708211104721274035250192240168079"
                ),
                MontFp!(
                    "19369486497309312061277881946237029141267272740526359302517213808446553098970"
                ),
                MontFp!(
                    "7940094733038062793529515395991096371773571922094772311764220189482319521850"
                ),
                MontFp!(
                    "26266119181754811201402200824861638328199794191715057380805332617547011668524"
                ),
            ],
            [
                MontFp!(
                    "20608074707185814339023032907318341048383060410959412048522855428266725599501"
                ),
                MontFp!(
                    "9178271014237970602770079696700157329530996359568103524699637399592478664026"
                ),
                MontFp!(
                    "47701107590146914868900666400919442563800477593292665707156993009143683237879"
                ),
                MontFp!(
                    "3510820734179449653960543969996663318747627903356941277690341076519268089320"
                ),
            ],
            [
                MontFp!(
                    "29348846719777431077739407126342807736758386871395967810152454975269135235218"
                ),
                MontFp!(
                    "21508842860285579109338877542083339527955542046411533438794905664670390294342"
                ),
                MontFp!(
                    "18680328723324263359036886083981681379132175831892967815891811931257462555218"
                ),
                MontFp!(
                    "50828231123138782878622888054010576359796238754328986008387427720372912792647"
                ),
            ],
            [
                MontFp!(
                    "28657794794998556716272629977514546250764013681125064867971574349218114625172"
                ),
                MontFp!(
                    "33079740383457282208302947956039568254422672391664354749193994044721067903799"
                ),
                MontFp!(
                    "37580317454617942481981990175878994567582554412855387702681533543699169440689"
                ),
                MontFp!(
                    "33018635415713911861411428294953314819462172257288917961589949901293277681705"
                ),
            ],
            [
                MontFp!(
                    "17524907224496794164960512560601728614079536896047942054509883695026269668154"
                ),
                MontFp!(
                    "46178517456006085277643641965935320508071532111249706454076466951452882061038"
                ),
                MontFp!(
                    "47488030983205427803034359403088928732610191936586392174026127541604111983497"
                ),
                MontFp!(
                    "12310986098205956955072933932049021648030702019638927472504623814447230877781"
                ),
            ],
            [
                MontFp!(
                    "34907433682181565642896658109600276718836563559731138130401593764879942750513"
                ),
                MontFp!(
                    "38440948138990743326152617869356648218121822747512371141022332627266826516600"
                ),
                MontFp!(
                    "20173089575902300256118122563311893974442028804411441716578684902893120439744"
                ),
                MontFp!(
                    "5355371110856184363013743931928911946383876366869179081262327334887335950591"
                ),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiJubjub_4_3::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
