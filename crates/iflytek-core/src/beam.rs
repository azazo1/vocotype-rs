use std::cmp::Ordering;

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug)]
pub struct BeamSearchConfig {
    pub beam_width: usize,
    pub input_score_count: usize,
    pub search_score_count: usize,
    pub merge_target_score_id: usize,
    pub eos_score_id: usize,
    pub score_floor: f32,
    pub step_divisor: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 8,
            input_score_count: 14_832,
            search_score_count: 14_831,
            merge_target_score_id: 14_828,
            eos_score_id: 14_830,
            score_floor: -20_000.0,
            step_divisor: 8,
        }
    }
}

impl BeamSearchConfig {
    pub fn validate(self) -> Result<Self> {
        if self.beam_width == 0
            || self.search_score_count == 0
            || self.search_score_count > self.input_score_count
            || self.beam_width > self.search_score_count
            || self.merge_target_score_id >= self.search_score_count
            || self.eos_score_id >= self.search_score_count
            || self.merge_target_score_id == self.eos_score_id
            || self.step_divisor == 0
        {
            bail!("invalid EdgeEsr beam search configuration")
        }
        Ok(self)
    }

    pub fn eos_token(self) -> i32 {
        self.eos_score_id as i32 + 1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeamCandidate {
    pub normalized_score: f32,
    pub acoustic_score: f32,
    pub parent: usize,
    pub token: i32,
    pub step: usize,
    pub path: Vec<i32>,
    pub terminal: bool,
}

#[derive(Clone, Debug)]
pub struct BeamSearchResult {
    pub candidates: Vec<BeamCandidate>,
    pub retained_finished: Vec<BeamCandidate>,
    pub ranked_candidates: Vec<BeamCandidate>,
    pub done: bool,
    pub first_eos_step: Option<usize>,
}

impl BeamSearchResult {
    pub fn parent_indices(&self) -> Vec<usize> {
        self.candidates.iter().map(|candidate| candidate.parent).collect()
    }

    pub fn token_ids(&self) -> Vec<i32> {
        self.candidates
            .iter()
            .map(|candidate| candidate.token - 1)
            .collect()
    }

    pub fn lengths(&self) -> Vec<i32> {
        self.candidates
            .iter()
            .map(|candidate| candidate.step as i32)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct OriginalBeamSearch {
    config: BeamSearchConfig,
    step_index: usize,
    active: Vec<BeamCandidate>,
    finished: Vec<BeamCandidate>,
    first_eos_step: Option<usize>,
    done: bool,
}

impl OriginalBeamSearch {
    pub fn new(config: BeamSearchConfig) -> Result<Self> {
        let config = config.validate()?;
        let mut search = Self {
            config,
            step_index: 0,
            active: Vec::new(),
            finished: Vec::new(),
            first_eos_step: None,
            done: false,
        };
        search.reset();
        Ok(search)
    }

    pub fn config(&self) -> BeamSearchConfig {
        self.config
    }

    pub fn reset(&mut self) {
        self.step_index = 0;
        self.active = vec![BeamCandidate {
            normalized_score: 0.0,
            acoustic_score: 0.0,
            parent: 0,
            token: 0,
            step: 0,
            path: Vec::new(),
            terminal: false,
        }];
        self.finished.clear();
        self.first_eos_step = None;
        self.done = false;
    }

    pub fn step(
        &mut self,
        scores: &[f32],
        postproc_flags: &[i32],
        frame_budget: usize,
    ) -> Result<BeamSearchResult> {
        if self.done {
            bail!("EdgeEsr beam search is already complete")
        }
        if scores.len() != self.active.len() * self.config.input_score_count
            || postproc_flags.len() != self.active.len()
        {
            bail!("EdgeEsr decoder score shape does not match active beams")
        }
        let processed = preprocess_scores(
            scores,
            postproc_flags,
            self.active.len(),
            self.config,
        )?;
        self.step_index += 1;
        let step = self.step_index;
        let max_steps = frame_budget.div_ceil(self.config.step_divisor);
        let mut candidates = Vec::with_capacity(self.active.len() * self.config.beam_width);
        for (parent, active) in self.active.iter().enumerate() {
            let row = &processed[parent * self.config.search_score_count
                ..(parent + 1) * self.config.search_score_count];
            let top = select_top_k(row, self.config.beam_width)?;
            if self.first_eos_step.is_none()
                && parent < 2
                && top.iter().take(2).any(|(_, token)| *token == self.config.eos_token())
            {
                self.first_eos_step = Some(step);
            }
            let previous_total = active.normalized_score * (step - 1) as f32;
            for (acoustic_score, token) in top {
                let normalized_score = (previous_total + acoustic_score) / step as f32;
                let terminal = token == self.config.eos_token() || step >= max_steps;
                let mut path = active.path.clone();
                path.push(token);
                candidates.push(BeamCandidate {
                    normalized_score,
                    acoustic_score,
                    parent,
                    token,
                    step,
                    path,
                    terminal,
                });
            }
        }
        rank_candidates(&mut candidates);
        let ranked_candidates = candidates.clone();
        candidates.extend(self.finished.iter().cloned());
        rank_candidates(&mut candidates);
        candidates.truncate(self.config.beam_width);
        let active = candidates
            .iter()
            .filter(|candidate| !candidate.terminal)
            .cloned()
            .collect::<Vec<_>>();
        let finished = candidates
            .iter()
            .filter(|candidate| candidate.terminal)
            .cloned()
            .collect::<Vec<_>>();
        self.active = active.clone();
        self.finished = finished.clone();
        self.done = active.is_empty();
        Ok(BeamSearchResult {
            candidates: active,
            retained_finished: finished,
            ranked_candidates,
            done: self.done,
            first_eos_step: self.first_eos_step,
        })
    }

    pub fn active(&self) -> &[BeamCandidate] {
        &self.active
    }

    pub fn finished(&self) -> &[BeamCandidate] {
        &self.finished
    }

    pub fn best_candidate(&self, finalized: bool) -> Option<&BeamCandidate> {
        if finalized || self.done {
            self.finished
                .first()
                .or_else(|| self.active.first().filter(|candidate| !candidate.path.is_empty()))
        } else {
            self.active
                .first()
                .filter(|candidate| !candidate.path.is_empty())
                .or_else(|| self.finished.first())
        }
    }

    pub fn step_index(&self) -> usize {
        self.step_index
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

pub fn preprocess_scores(
    scores: &[f32],
    postproc_flags: &[i32],
    beam_count: usize,
    config: BeamSearchConfig,
) -> Result<Vec<f32>> {
    if scores.len() != beam_count * config.input_score_count
        || postproc_flags.len() != beam_count
    {
        bail!("EdgeEsr score preprocessing shape mismatch")
    }
    let mut output = vec![0.0; beam_count * config.search_score_count];
    for beam in 0..beam_count {
        let flag = postproc_flags[beam];
        if !(0..=3).contains(&flag) {
            bail!("EdgeEsr postprocess flag uses unsupported bits")
        }
        let source = &scores[beam * config.input_score_count
            ..(beam + 1) * config.input_score_count];
        let target = &mut output[beam * config.search_score_count
            ..(beam + 1) * config.search_score_count];
        target.copy_from_slice(&source[..config.search_score_count]);
        if flag & 1 != 0 {
            target.fill(config.score_floor);
            target[config.merge_target_score_id] = 0.0;
            target[config.eos_score_id] = source[config.eos_score_id];
        }
        if flag & 2 == 0 {
            let left = source[config.merge_target_score_id].exp();
            let right = source[config.eos_score_id].exp();
            target[config.merge_target_score_id] = (left + right).ln();
            target[config.eos_score_id] = config.score_floor;
        }
    }
    Ok(output)
}

pub fn select_top_k(scores: &[f32], count: usize) -> Result<Vec<(f32, i32)>> {
    if count == 0 || count > scores.len() || scores.iter().any(|score| !score.is_finite()) {
        bail!("invalid EdgeEsr top-k input")
    }
    let mut heap = Vec::with_capacity(count + 1);
    for (score_id, score) in scores.iter().copied().enumerate() {
        let item = (score, score_id);
        if heap.len() < count {
            heap_push_min(&mut heap, item);
        } else if heap[0].0 < item.0 {
            heap_push_min(&mut heap, item);
            let _ = heap_pop_min(&mut heap);
        }
    }
    let mut ascending = Vec::with_capacity(count);
    while !heap.is_empty() {
        ascending.push(heap_pop_min(&mut heap));
    }
    ascending.reverse();
    Ok(ascending
        .into_iter()
        .map(|(score, score_id)| (score, score_id as i32 + 1))
        .collect())
}

fn heap_push_min(heap: &mut Vec<(f32, usize)>, item: (f32, usize)) {
    heap.push(item);
    let mut child = heap.len() - 1;
    while child > 0 {
        let parent = (child - 1) / 2;
        if item.0 >= heap[parent].0 {
            break;
        }
        heap[child] = heap[parent];
        child = parent;
    }
    heap[child] = item;
}

fn heap_pop_min(heap: &mut Vec<(f32, usize)>) -> (f32, usize) {
    let result = heap[0];
    let last = heap.pop().expect("heap is not empty");
    let size = heap.len();
    if size == 0 {
        return result;
    }
    if size == 1 {
        heap[0] = last;
        return result;
    }
    let mut child = 1;
    if size >= 3 && heap[1].0 > heap[2].0 {
        child = 2;
    }
    if heap[child].0 > last.0 {
        heap[0] = last;
        return result;
    }
    let mut parent = 0;
    loop {
        heap[parent] = heap[child];
        parent = child;
        child = parent * 2 + 1;
        if child >= size {
            break;
        }
        if child + 1 < size && heap[child].0 > heap[child + 1].0 {
            child += 1;
        }
        if heap[child].0 > last.0 {
            break;
        }
    }
    heap[parent] = last;
    result
}

fn rank_candidates(candidates: &mut [BeamCandidate]) {
    candidates.sort_by(|left, right| {
        right
            .normalized_score
            .partial_cmp(&left.normalized_score)
            .unwrap_or(Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::{BeamSearchConfig, OriginalBeamSearch, preprocess_scores, select_top_k};

    fn config() -> BeamSearchConfig {
        BeamSearchConfig {
            beam_width: 2,
            input_score_count: 6,
            search_score_count: 5,
            merge_target_score_id: 3,
            eos_score_id: 4,
            score_floor: -20_000.0,
            step_divisor: 2,
        }
    }

    #[test]
    fn preprocessing_handles_all_flag_modes() {
        let scores = vec![
            -5.0, -4.0, -3.0, -2.0, -1.0, 99.0, -6.0, -5.0, -4.0, -3.0, -2.0,
            98.0, -7.0, -6.0, -5.0, -4.0, -3.0, 97.0, -8.0, -7.0, -6.0, -5.0,
            -4.0, 96.0,
        ];
        let output = preprocess_scores(&scores, &[0, 1, 2, 3], 4, config()).expect("scores");
        assert_eq!(output[4], -20_000.0);
        assert_eq!(output[5..8], [-20_000.0; 3]);
        assert_eq!(&output[10..15], &scores[12..17]);
        assert_eq!(&output[15..20], &[-20_000.0, -20_000.0, -20_000.0, 0.0, -4.0]);
    }

    #[test]
    fn top_k_preserves_original_floor_heap_order() {
        let mut scores = vec![-20_000.0; 9];
        scores[8] = -3.5;
        let values = select_top_k(&scores, 8).expect("top k");
        let tokens = values.into_iter().map(|(_, token)| token).collect::<Vec<_>>();
        assert_eq!(tokens, [9, 3, 6, 7, 5, 8, 4, 2]);
    }

    #[test]
    fn search_keeps_parent_indices_and_normalizes_scores() {
        let mut search = OriginalBeamSearch::new(config()).expect("search");
        let first = search
            .step(&[-0.1, -0.2, -5.0, -6.0, -7.0, 0.0], &[2], 20)
            .expect("first");
        assert_eq!(first.token_ids(), [0, 1]);
        let second = search
            .step(
                &[
                    -0.5, -3.0, -4.0, -5.0, -6.0, 0.0, -0.1, -3.0, -4.0, -5.0,
                    -6.0, 0.0,
                ],
                &[2, 2],
                20,
            )
            .expect("second");
        assert_eq!(second.parent_indices(), [1, 0]);
        assert!((second.candidates[0].normalized_score + 0.15).abs() < 1.0e-6);
    }
}
