use cfg_if::cfg_if;
use std::convert::TryFrom;

#[cfg(feature = "rand")]
use rand::Rng;

/// A side, or face, of a die.
/// element is a unique id for the face.
/// weight contributes to the chance of this
/// face being selected, compared to others.
/// R weight of 0 means the face won't ever be rolled.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct WeightedSide {
    pub element: u64,
    pub weight: u32,
}

/// This is a weighted die. You can add sides (faces),
/// change their weights, and so on.
#[derive(Clone, Eq, PartialEq)]
pub struct WeightedDie {
    /// An element and its probabalistic weight,
    /// compared with its peers.
    items: Vec<WeightedSide>,

    /// Caching running weights in order to support
    /// O(lg n) rolls.
    running_weight: Vec<u64>,
}

impl WeightedDie {
    /// Create a new weighted die with the given sides (faces).
    /// Runs in O(n).
    pub fn new(items: Vec<WeightedSide>) -> Self {
        let mut tmp = WeightedDie {
            items: items,
            running_weight: vec![],
        };
        tmp.update_running_weights();
        tmp
    }

    fn find_first(&mut self, element: u64) -> Option<(usize, &mut WeightedSide)> {
        let found_val = self
            .items
            .iter_mut()
            .enumerate()
            .filter(|v| v.1.element == element)
            .next();
        found_val
    }

    fn gcd(a: u64, b: u64) -> u64 {
        let mut a = a;
        let mut b = b;
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    fn less_lossy_divide(numerator: u64, denominator: u64) -> f32 {
        let mut n = numerator;
        let mut d = denominator;
        let gcd = Self::gcd(n, d);
        n = n / gcd;
        d = d / gcd;
        return n as f32 / d as f32;
    }

    /// Returns the probability of rolling the selected side.
    /// If the die does not contain the side, returns 0.
    /// This will be off since floats aren't exact sometimes.
    pub fn get_probability(&mut self, element: u64) -> f32 {
        match self.find_first(element) {
            Some(v) => {
                // if there is some found element, then
                // running_weight is not empty.
                Self::less_lossy_divide(
                    v.1.weight as u64,
                    *self.running_weight.last().unwrap_or(&1),
                )
            }
            None => 0.0,
        }
    }

    fn update_running_weights(&mut self) {
        self.running_weight = self
            .items
            .iter()
            .scan(0 as u64, |state, &x| {
                *state += x.weight as u64;
                Some(*state)
            })
            .collect();
    }

    pub fn sides(&self) -> Vec<WeightedSide> {
        self.items.clone()
    }

    /// Removes and returns the given side, if it exists.
    pub fn remove(&mut self, elem: u64) -> Option<WeightedSide> {
        let found = self.find_first(elem);
        match found {
            Some(v) => {
                let pop_idx = v.0;
                let side = self.items.remove(pop_idx);
                self.update_running_weights();
                Some(side)
            }
            None => None,
        }
    }

    /// Modifies the weight of an element in the collection.
    /// If it doesn't exist, will add to the collection.
    /// If the new weight would be less than 0, removes
    /// the element from the collection.
    /// Runs in O(n).
    pub fn modify(&mut self, elem: u64, weight_delta: i32) {
        let found = self.find_first(elem);
        let abs_delta_try = u32::try_from(weight_delta.abs()).ok();
        match found {
            Some(v) => {
                // In the collction, so modify it.
                match abs_delta_try {
                    Some(abs_delta) => {
                        if weight_delta > 0 {
                            // the delta is positive. simple case.
                            v.1.weight += abs_delta;
                        } else if abs_delta >= v.1.weight {
                            // the weight delta is negative and really big
                            let pop_idx = v.0;
                            self.items.remove(pop_idx);
                        } else {
                            // the weight is negative, but not big enough to
                            // remove the item.
                            v.1.weight -= abs_delta;
                        }
                    }
                    None => panic!("abs(i32) can't fit into u64?!"),
                }
            }
            None => {
                // Not in the collection, so add it.
                if weight_delta > 0 {
                    match abs_delta_try {
                        Some(abs_delta) => {
                            self.items.push(WeightedSide {
                                element: elem,
                                weight: abs_delta,
                            });
                        }
                        None => panic!("abs(i32) can't fit into u64?!"),
                    }
                } else {
                    // nothing to do at all
                }
            }
        }

        // now update running weights
        self.update_running_weights();
    }

    /// Select some element from the collection.
    /// This doesn't remove the element.
    /// roll is an optional param when you don't want
    /// to rely on a random value.
    /// Runs in O(lg n).
    pub fn roll(&self, roll: Option<u64>) -> Option<u64> {
        // If there is nothing to roll, return nothing.
        if self.items.len() == 0 {
            return None;
        }

        // Figure out the roll value, if supplied.
        let total_weight = *self.running_weight.last().unwrap_or(&0);
        let roll_result: u64 = match roll {
            Some(r) => r % total_weight,
            None => {
                cfg_if! {
                    if #[cfg(feature = "rand")] {
                        let mut rng = rand::thread_rng();
                        rng.gen_range(0, total_weight) as u64
                    } else {
                        panic!("'roll' param is not optional when the 'rand' feature is off.");
                    }
                }
            }
        };

        // Binary search for the matching element.
        let mut start: usize = 0;
        let mut end: usize = self.items.len() - 1;
        while start <= end {
            let mid = (end + start) / 2;
            let matched = self.running_weight[mid];

            let mut one_less: u64 = 0;
            if mid > 0 {
                one_less = self.running_weight[mid - 1];
            }

            if matched > roll_result {
                if one_less <= roll_result {
                    // lt current element, but gte than
                    // the next smallest = we got our match.
                    return Some(self.items[mid].element);
                } else {
                    // further to the left
                    end = mid - 1;
                }
            } else {
                // further to the right
                start = mid + 1;
            }
        }

        return Some(self.items[start].element);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let c = WeightedDie::new(vec![]);
        assert_eq!(c.roll(Some(0)), None);
    }

    #[test]
    fn one_from_constructor() {
        let c = WeightedDie::new(vec![WeightedSide {
            element: 99,
            weight: 3,
        }]);
        assert_eq!(c.roll(Some(0)), Some(99));
        assert_eq!(c.roll(Some(1)), Some(99));
        assert_eq!(c.roll(Some(2)), Some(99));
    }

    #[test]
    fn one_from_insertion() {
        let mut c = WeightedDie::new(vec![]);
        c.modify(99, 3);
        assert_eq!(c.roll(Some(0)), Some(99));
        assert_eq!(c.roll(Some(1)), Some(99));
        assert_eq!(c.roll(Some(2)), Some(99));
    }

    #[test]
    fn coin() {
        let mut c = WeightedDie::new(vec![]);
        c.modify(1, 1);
        c.modify(2, 1);

        assert_eq!(c.items.len(), 2);

        assert_eq!(c.roll(Some(0)), Some(1));
        assert_eq!(c.roll(Some(1)), Some(2));
        assert_eq!(c.roll(Some(2)), Some(1)); // rolled over
    }

    #[test]
    fn six_sided_die() {
        let mut c = WeightedDie::new(vec![]);
        c.modify(1, 1);
        c.modify(2, 1);
        c.modify(3, 1);
        c.modify(4, 1);
        c.modify(5, 1);
        c.modify(6, 1);

        assert_eq!(c.items.len(), 6);

        assert_eq!(c.roll(Some(0)), Some(1));
        assert_eq!(c.roll(Some(1)), Some(2));
        assert_eq!(c.roll(Some(2)), Some(3));
        assert_eq!(c.roll(Some(3)), Some(4));
        assert_eq!(c.roll(Some(4)), Some(5));
        assert_eq!(c.roll(Some(5)), Some(6));
        assert_eq!(c.roll(Some(6)), Some(1)); // rolled over
    }

    #[test]
    fn coin_with_edge() {
        let mut c = WeightedDie::new(vec![]);
        c.modify(1, 100);
        c.modify(2, 1);
        c.modify(3, 100);

        assert_eq!(c.items.len(), 3);

        assert_eq!(c.roll(Some(0)), Some(1));
        assert_eq!(c.roll(Some(98)), Some(1));
        assert_eq!(c.roll(Some(99)), Some(1));
        assert_eq!(c.roll(Some(100)), Some(2));
        assert_eq!(c.roll(Some(101)), Some(3));
        assert_eq!(c.roll(Some(200)), Some(3));
        assert_eq!(c.roll(Some(230)), Some(1)); // rolled over
    }

    #[test]
    fn modified_weight() {
        let mut c = WeightedDie::new(vec![]);
        c.modify(1, 10);
        c.modify(2, 10);
        c.modify(1, 10);
        c.modify(3, 10);
        c.modify(2, -5);
        c.modify(3, -10);
        // at this point, we have (1, 20) and (2, 5).

        assert_eq!(c.items.len(), 2);

        assert_eq!(c.roll(Some(0)), Some(1));
        assert_eq!(c.roll(Some(9)), Some(1));
        assert_eq!(c.roll(Some(19)), Some(1));
        assert_eq!(c.roll(Some(20)), Some(2));
        assert_eq!(c.roll(Some(29)), Some(1)); // rolled over
    }

    #[test]
    fn get_prob() {
        let mut c = WeightedDie::new(vec![]);
        c.modify(1, 10);
        c.modify(2, 10);
        c.modify(1, 10);
        c.modify(3, 10);
        c.modify(2, -5);
        c.modify(3, -10);
        // at this point, we have (1, 20) and (2, 5).

        assert_eq!(c.items.len(), 2);

        let tmp = c.get_probability(1);
        assert_eq!(tmp, 20.0 / 25.0);
        assert_eq!(c.get_probability(2), 5.0 / 25.0);
        assert_eq!(c.get_probability(9), 0.0);
    }
}
