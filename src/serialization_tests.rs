extern crate ron;

use crate::Element;
use crate::MarkovChain;

pub fn assert_chain_eq<E>(a: &MarkovChain<E>, b: &MarkovChain<E>)
where
    E: Element,
{
    assert_eq!(a, b);

    let a_s = ron::to_string(a).unwrap();
    let b_s = ron::to_string(b).unwrap();

    let a_de: MarkovChain<E> = ron::from_str(&a_s).unwrap();
    let b_de: MarkovChain<E> = ron::from_str(&b_s).unwrap();

    assert_eq!(a_de, b_de);
}

#[test]
fn same_train_order_eq() {
    let mut m1 = MarkovChain::new(1, &[]);
    let mut m2 = MarkovChain::new(1, &[]);
    let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
    for i in 1..alpha.len() {
        m1.train(&[alpha[i - 1]], alpha[i], 1);
        m2.train(&[alpha[i - 1]], alpha[i], 1);
    }
    assert_chain_eq(&m1, &m2);
}

#[test]
fn diff_train_order_eq() {
    let mut m1 = MarkovChain::new(1, &[]);
    let mut m2 = MarkovChain::new(1, &[]);
    let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
    for i in 1..alpha.len() {
        m1.train(&[alpha[i - 1]], alpha[i], 1);
        let rev = alpha.len() - i;
        m2.train(&[alpha[rev - 1]], alpha[rev], 1);
    }
    assert_chain_eq(&m1, &m2);
}
