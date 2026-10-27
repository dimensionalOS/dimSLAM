//! The newest of every tf edge seen so far, looked up across chains in either direction.

use std::collections::{HashMap, HashSet, VecDeque};

use dim_slam::nalgebra::Isometry3;
use dim_slam::TfLookup;

pub struct TfBuffer {
    /// (parent, child) -> parent_from_child.
    edges: HashMap<(String, String), Isometry3<f64>>,
    /// Edges whose parent is one of these are dropped: they are poses the replay must not
    /// mistake for mounts (e.g. a recorded `odom -> base_link`).
    ignored_parents: HashSet<String>,
}

impl TfBuffer {
    pub fn ignoring(parents: &[String]) -> Self {
        Self {
            edges: HashMap::new(),
            ignored_parents: parents.iter().cloned().collect(),
        }
    }

    pub fn insert(&mut self, parent: String, child: String, transform: Isometry3<f64>) {
        if !self.ignored_parents.contains(&parent) {
            self.edges.insert((parent, child), transform);
        }
    }
}

impl TfLookup for TfBuffer {
    fn latest(&self, parent: &str, child: &str) -> Option<Isometry3<f64>> {
        if parent == child {
            return Some(Isometry3::identity());
        }
        // Breadth-first from the parent; each step carries parent_from_frame.
        let mut seen = HashSet::from([parent.to_string()]);
        let mut queue = VecDeque::from([(parent.to_string(), Isometry3::identity())]);
        while let Some((frame, parent_from_frame)) = queue.pop_front() {
            for ((a, b), a_from_b) in &self.edges {
                let next = if *a == frame {
                    Some((b, parent_from_frame * a_from_b))
                } else if *b == frame {
                    Some((a, parent_from_frame * a_from_b.inverse()))
                } else {
                    None
                };
                if let Some((next, parent_from_next)) = next {
                    if next == child {
                        return Some(parent_from_next);
                    }
                    if seen.insert(next.clone()) {
                        queue.push_back((next.clone(), parent_from_next));
                    }
                }
            }
        }
        None
    }
}
