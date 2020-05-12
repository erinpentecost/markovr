use super::die;

use std::collections::HashMap;
use std::convert::TryFrom;

/// Variable-order Markov chain.
pub struct MarkovChain {
    // the 'memory' for the MarkovChain chain.
    // 1 is the typical MarkovChain chain that only looks
    // back 1 element.
    // 0 is functionally equivalent to a weighteddie.
    order: usize,
    // the number of elements in the key should
    // exactly equal the order of the MarkovChain chain.
    probability_map: HashMap<Vec<u64>, die::WeightedDie>,
}

impl MarkovChain {
    /// Creates a new MarkovChain.
    ///
    /// 'order' is the order of the Markov Chain.
    ///
    ///  A value of 1 is your typical Markov Chain,
    /// that only looks back one place.
    ///
    /// A value of 0 is just a weighted die, since
    /// there is no memory.
    ///
    /// Higher-order Markov Chains are the building
    /// blocks of the waveform collapse algorithm.
    pub fn new(order: usize) -> Self {
        MarkovChain {
            order,
            probability_map: HashMap::new(),
        }
    }

    fn to_key(order: usize, view: &[u64]) -> Vec<u64> {
        view.into_iter()
            .skip(view.len() - order)
            .take(order)
            .cloned()
            .collect()
    }

    /// Feeds training data into the model.
    ///
    /// 'view' is the sliding window of elements to load
    /// into the MarkovChain chain. the number of elements in
    /// view should be self.order + 1 (excess will be ignored).
    /// the last element
    /// in view is the element to increase the weight of.
    ///
    /// 'weight_delta' should be the number of times we're
    /// loading this view into the model (typically 1 at
    /// a time).
    pub fn train(&mut self, view: &[u64], result: u64, weight_delta: i32) {
        let key = MarkovChain::to_key(self.order, view);

        self.probability_map
            .entry(key)
            .and_modify(|d| {
                d.modify(result, weight_delta);
            })
            .or_insert((|| match u32::try_from(weight_delta).ok() {
                Some(v) => die::WeightedDie::new(vec![die::WeightedSide {
                    element: result,
                    weight: v,
                }]),
                None => die::WeightedDie::new(vec![]),
            })());
    }

    /// Generates the next value, given the previous item(s).
    ///
    /// view is the sliding window of the latest elements.
    /// only the last self.order elements are looked at.
    ///
    /// roll allows for a deterministic result, if supplied.
    pub fn generate(&self, view: &[u64], roll: Option<u64>) -> Option<u64> {
        let key = MarkovChain::to_key(self.order, view);

        match self.probability_map.get(&key) {
            Some(v) => v.roll(roll),
            None => None,
        }
    }

    /// Returns the probability of getting 'result', given
    /// 'view'.
    pub fn probability(&self, view: &[u64], result: u64) -> f32 {
        let key = MarkovChain::to_key(self.order, view);

        let map = self.probability_map.get_key_value(&key);
        match map {
            Some(v) => v.1.get_probability(result),
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let m0 = MarkovChain::new(0);
        assert_eq!(m0.generate(&[], None), None);
        assert_eq!(m0.generate(&[1], None), None);
        assert_eq!(m0.generate(&[], Some(33)), None);

        let m1 = MarkovChain::new(1);
        assert_eq!(m1.generate(&[1], None), None);
        assert_eq!(m1.generate(&[1, 1], None), None);
        assert_eq!(m1.generate(&[1], Some(33)), None);

        let m2 = MarkovChain::new(2);
        assert_eq!(m2.generate(&[1, 1], None), None);
        assert_eq!(m2.generate(&[1, 1, 1], None), None);
        assert_eq!(m2.generate(&[1, 1], Some(33)), None);
    }

    #[test]
    fn alphabet_first_order() {
        let mut m = MarkovChain::new(1);

        // this could have just been a number range,
        // but it serves as an example of how to encode
        // an alphabet
        let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        let encoded: Vec<u64> = alpha
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, _x)| i as u64)
            .collect();

        for i in m.order..encoded.len() {
            m.train(&[encoded[i - 1]], encoded[i], 1);
        }

        for i in 0..(encoded.len() - 1) {
            let next = m.generate(&[encoded[i]].clone(), None);
            match next {
                Some(v) => assert_eq!(v, encoded[i + 1]),
                None => panic!(
                    "can't predict next letter after {} (encoded as {})",
                    alpha[i], encoded[i]
                ),
            };
        }
    }

    #[test]
    fn alphabet_second_order() {
        let mut m = MarkovChain::new(2);

        // this could have just been a number range,
        // but it serves as an example of how to encode
        // an alphabet
        let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        let encoded: Vec<u64> = alpha
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, _x)| i as u64)
            .collect();

        for i in m.order..encoded.len() {
            m.train(&[encoded[i - 2], encoded[i - 1]], encoded[i], 1);
        }

        for i in 1..(encoded.len() - 1) {
            let next = m.generate(&[encoded[i - 1], encoded[i].clone()], None);
            match next {
                Some(v) => assert_eq!(v, encoded[i + 1]),
                None => panic!(
                    "can't predict next letter after {} (encoded as {})",
                    alpha[i], encoded[i]
                ),
            };
        }
    }
}
