use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Result, bail};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;

use crate::{
    punctuation_context, punctuation_qk, punctuation_quantized_linear, punctuation_softmax,
};

pub(crate) const NUMERIC_LABELS: [&str; 19] = [
    "O", "SA", "BA", "MA", "EA", "SB", "BB", "MB", "EB", "SC", "BC", "MC",
    "EC", "SD", "BD", "MD", "ED", "E", "<unk>",
];

#[derive(Clone, Debug)]
enum Weight {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I8 { shape: Vec<usize>, data: Vec<i8> },
}

#[derive(Clone, Debug)]
struct WeightArchive {
    values: HashMap<String, Weight>,
}

impl WeightArchive {
    fn load(path: &Path, strip_arg_prefix: bool) -> Result<Self> {
        let stream = File::open(path)?;
        let mut archive = NpzReader::new(BufReader::new(stream))?;
        let names = archive.names()?;
        let mut values = HashMap::with_capacity(names.len());
        for name in names {
            let key = if strip_arg_prefix {
                name.strip_prefix("arg:").unwrap_or(&name).to_string()
            } else {
                name.clone()
            };
            if let Ok(array) = archive.by_name::<ndarray::OwnedRepr<f32>, ndarray::IxDyn>(&name) {
                let shape = array.shape().to_vec();
                let (data, offset) = array.into_raw_vec_and_offset();
                if offset.unwrap_or(0) != 0 {
                    bail!("postprocess weight {} is not contiguous", name)
                }
                values.insert(key, Weight::F32 { shape, data });
                continue;
            }
            let array: ArrayD<i8> = archive.by_name(&name).map_err(|error| {
                anyhow::anyhow!("postprocess weight {} has unsupported dtype: {}", name, error)
            })?;
            let shape = array.shape().to_vec();
            let (data, offset) = array.into_raw_vec_and_offset();
            if offset.unwrap_or(0) != 0 {
                bail!("postprocess weight {} is not contiguous", name)
            }
            values.insert(key, Weight::I8 { shape, data });
        }
        Ok(Self { values })
    }

    fn f32(&self, name: &str) -> Result<(&[usize], &[f32])> {
        match self.values.get(name) {
            Some(Weight::F32 { shape, data }) => Ok((shape, data)),
            Some(Weight::I8 { .. }) => bail!("postprocess weight {} must use f32", name),
            None => bail!("postprocess model is missing weight {}", name),
        }
    }

    fn i8(&self, name: &str) -> Result<(&[usize], &[i8])> {
        match self.values.get(name) {
            Some(Weight::I8 { shape, data }) => Ok((shape, data)),
            Some(Weight::F32 { .. }) => bail!("postprocess weight {} must use i8", name),
            None => bail!("postprocess model is missing weight {}", name),
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NumericModel {
    weights: WeightArchive,
    token_to_id: HashMap<String, usize>,
    quantized_weights: HashMap<String, Vec<i16>>,
}

impl NumericModel {
    const HIDDEN_SIZE: usize = 128;
    const HEAD_COUNT: usize = 8;
    const LAYER_COUNT: usize = 5;
    const PROJECTION_SIZE: usize = 32;
    pub(crate) const MAX_LENGTH: usize = 512;

    pub(crate) fn load(weights: &Path, vocabulary: &Path) -> Result<Self> {
        let weights = WeightArchive::load(weights, false)?;
        let vocabulary = std::fs::read_to_string(vocabulary)?
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let token_to_id = vocabulary
            .iter()
            .enumerate()
            .map(|(index, token)| (token.clone(), index))
            .collect::<HashMap<_, _>>();
        if !token_to_id.contains_key("<unk>") {
            bail!("numeric vocabulary does not contain <unk>")
        }
        let (embedding_shape, _) = weights.f32("embeddings")?;
        if embedding_shape.first().copied() != Some(vocabulary.len()) {
            bail!("numeric vocabulary and embedding rows differ")
        }
        let mut quantized_weights = HashMap::new();
        for layer in 0..Self::LAYER_COUNT {
            for array in ["00", "02", "06"] {
                let name = format!("layer_{}_array_{}", layer, array);
                let (_, values) = weights.f32(&name)?;
                quantized_weights.insert(name, quantize_i16(values));
            }
        }
        Ok(Self {
            weights,
            token_to_id,
            quantized_weights,
        })
    }

    pub(crate) fn predict(&self, tokens: &[String], features: &[f32]) -> Result<Vec<String>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if tokens.len() > Self::MAX_LENGTH || features.len() != tokens.len() * 2 {
            bail!("numeric model input shape is invalid")
        }
        let mut hidden = vec![0.0; tokens.len() * Self::HIDDEN_SIZE];
        let unknown = self.token_to_id["<unk>"];
        let (_, embeddings) = self.weights.f32("embeddings")?;
        for (row, token) in tokens.iter().enumerate() {
            let token_id = self.token_to_id.get(token).copied().unwrap_or(unknown);
            let source = token_id * Self::HIDDEN_SIZE;
            hidden[row * Self::HIDDEN_SIZE..(row + 1) * Self::HIDDEN_SIZE]
                .copy_from_slice(&embeddings[source..source + Self::HIDDEN_SIZE]);
            hidden[row * Self::HIDDEN_SIZE + 126] +=
                features[row * 2] * (Self::HIDDEN_SIZE as f32).sqrt();
            hidden[row * Self::HIDDEN_SIZE + 127] +=
                features[row * 2 + 1] * (Self::HIDDEN_SIZE as f32).sqrt();
        }
        add_positional_encoding(&mut hidden, tokens.len(), Self::HIDDEN_SIZE);
        let head_width = Self::HIDDEN_SIZE / Self::HEAD_COUNT;
        for layer in 0..Self::LAYER_COUNT {
            let prefix = format!("layer_{}_array_", layer);
            let mut feed_forward = self.quantized_linear(
                &hidden,
                tokens.len(),
                Self::HIDDEN_SIZE,
                &(prefix.clone() + "00"),
                &(prefix.clone() + "01"),
            )?;
            for value in &mut feed_forward {
                *value = value.max(0.0);
            }
            feed_forward = self.quantized_linear(
                &feed_forward,
                tokens.len(),
                feed_forward.len() / tokens.len(),
                &(prefix.clone() + "02"),
                &(prefix.clone() + "03"),
            )?;
            add_in_place(&mut hidden, &feed_forward)?;
            hidden = layer_norm(
                &hidden,
                tokens.len(),
                Self::HIDDEN_SIZE,
                self.weights.f32(&(prefix.clone() + "04"))?.1,
                self.weights.f32(&(prefix.clone() + "05"))?.1,
                1.0e-6,
            )?;

            let qkv = linear(
                &hidden,
                tokens.len(),
                Self::HIDDEN_SIZE,
                self.weights.f32(&(prefix.clone() + "08"))?.1,
                self.weights.f32(&(prefix.clone() + "09"))?.1,
            )?;
            let mut context = vec![0.0; tokens.len() * Self::HIDDEN_SIZE];
            for head in 0..Self::HEAD_COUNT {
                let mut scores = vec![0.0; tokens.len() * tokens.len()];
                for row in 0..tokens.len() {
                    for column in 0..tokens.len() {
                        let mut sum = 0.0;
                        for channel in 0..head_width {
                            let q = qkv[row * Self::HIDDEN_SIZE * 3
                                + head * head_width
                                + channel];
                            let k = qkv[column * Self::HIDDEN_SIZE * 3
                                + Self::HIDDEN_SIZE
                                + head * head_width
                                + channel];
                            sum += q * k;
                        }
                        scores[row * tokens.len() + column] =
                            sum * (1.0 / (head_width as f32).sqrt());
                    }
                }
                let probabilities = softmax_rows(&scores, tokens.len(), tokens.len())?;
                for row in 0..tokens.len() {
                    for channel in 0..head_width {
                        let mut value = 0.0;
                        for column in 0..tokens.len() {
                            value += probabilities[row * tokens.len() + column]
                                * qkv[column * Self::HIDDEN_SIZE * 3
                                    + Self::HIDDEN_SIZE * 2
                                    + head * head_width
                                    + channel];
                        }
                        context[row * Self::HIDDEN_SIZE + head * head_width + channel] = value;
                    }
                }
            }
            let attention_output = self.quantized_linear(
                &context,
                tokens.len(),
                Self::HIDDEN_SIZE,
                &(prefix.clone() + "06"),
                &(prefix.clone() + "07"),
            )?;
            add_in_place(&mut hidden, &attention_output)?;
            hidden = layer_norm(
                &hidden,
                tokens.len(),
                Self::HIDDEN_SIZE,
                self.weights.f32(&(prefix.clone() + "10"))?.1,
                self.weights.f32(&(prefix + "11"))?.1,
                1.0e-6,
            )?;
        }

        let projected = linear(
            &hidden,
            tokens.len(),
            Self::HIDDEN_SIZE,
            self.weights.f32("projection_weight")?.1,
            self.weights.f32("projection_bias")?.1,
        )?;
        let (_, label_offsets) = self.weights.f32("label_offsets")?;
        let (_, label_vector) = self.weights.f32("label_vector")?;
        let (_, label_context) = self.weights.f32("label_context")?;
        let label_scalar = self.weights.f32("label_scalar")?.1[0];
        let mut label_logits = vec![0.0; tokens.len() * NUMERIC_LABELS.len()];
        for row in 0..tokens.len() {
            for label in 0..NUMERIC_LABELS.len() {
                let mut value = 0.0;
                for channel in 0..Self::PROJECTION_SIZE {
                    let transformed = native_tanh(
                        projected[row * Self::PROJECTION_SIZE + channel]
                            + label_offsets[label * Self::PROJECTION_SIZE + channel],
                    );
                    value += transformed * label_vector[channel];
                }
                label_logits[row * NUMERIC_LABELS.len() + label] = value + label_scalar;
            }
        }
        let label_probabilities = softmax_rows(
            &label_logits,
            tokens.len(),
            NUMERIC_LABELS.len(),
        )?;
        let mut label_features = vec![0.0; tokens.len() * Self::HIDDEN_SIZE];
        for row in 0..tokens.len() {
            for channel in 0..Self::HIDDEN_SIZE {
                for label in 0..NUMERIC_LABELS.len() {
                    label_features[row * Self::HIDDEN_SIZE + channel] +=
                        label_probabilities[row * NUMERIC_LABELS.len() + label]
                            * label_context[label * Self::HIDDEN_SIZE + channel];
                }
            }
        }
        let mut gate_input = Vec::with_capacity(tokens.len() * Self::HIDDEN_SIZE * 2);
        for row in 0..tokens.len() {
            gate_input.extend_from_slice(
                &hidden[row * Self::HIDDEN_SIZE..(row + 1) * Self::HIDDEN_SIZE],
            );
            gate_input.extend_from_slice(
                &label_features[row * Self::HIDDEN_SIZE..(row + 1) * Self::HIDDEN_SIZE],
            );
        }
        let gate_logits = linear(
            &gate_input,
            tokens.len(),
            Self::HIDDEN_SIZE * 2,
            self.weights.f32("gate_weight")?.1,
            self.weights.f32("gate_bias")?.1,
        )?;
        let mut blended = vec![0.0; hidden.len()];
        for index in 0..blended.len() {
            let gate = 1.0 / (1.0 + (-gate_logits[index].abs()).exp());
            blended[index] = hidden[index] * gate + label_features[index] * (1.0 - gate);
        }
        let logits = linear(
            &blended,
            tokens.len(),
            Self::HIDDEN_SIZE,
            self.weights.f32("classifier_weight")?.1,
            self.weights.f32("classifier_bias")?.1,
        )?;
        let probabilities = softmax_rows(&logits, tokens.len(), NUMERIC_LABELS.len())?;
        Ok((0..tokens.len())
            .map(|row| {
                let label = argmax(
                    &probabilities[row * NUMERIC_LABELS.len()
                        ..(row + 1) * NUMERIC_LABELS.len()],
                );
                NUMERIC_LABELS[label].to_string()
            })
            .collect())
    }

    fn quantized_linear(
        &self,
        input: &[f32],
        rows: usize,
        inner: usize,
        weight_name: &str,
        bias_name: &str,
    ) -> Result<Vec<f32>> {
        let bias = self.weights.f32(bias_name)?.1;
        let columns = bias.len();
        let weight = self
            .quantized_weights
            .get(weight_name)
            .ok_or_else(|| anyhow::anyhow!("numeric model is missing {}", weight_name))?;
        if input.len() != rows * inner || weight.len() != columns * inner {
            bail!("numeric quantized linear shape mismatch")
        }
        let quantized_input = quantize_i16(input);
        let mut output = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut accumulator = 0_i32;
                for index in 0..inner {
                    accumulator += i32::from(quantized_input[row * inner + index])
                        * i32::from(weight[column * inner + index]);
                }
                output[row * columns + column] =
                    accumulator as f32 * (1.0 / (1024.0 * 1024.0)) + bias[column];
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct PunctuationMap {
    entries: Vec<(Vec<u8>, Vec<String>)>,
}

impl PunctuationMap {
    fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        const MAGIC: &[u8] = b"ANIONEXPMAP1";
        if !data.starts_with(MAGIC) || data.len() < MAGIC.len() + 4 {
            bail!("invalid punctuation maplist magic")
        }
        let count = read_u32(&data, MAGIC.len())? as usize;
        let offsets_start = MAGIC.len() + 4;
        let data_start = offsets_start + (count + 1) * 4;
        if data_start > data.len() || read_u32(&data, offsets_start)? != 0 {
            bail!("invalid punctuation maplist offsets")
        }
        let mut entries = Vec::with_capacity(count);
        for index in 0..count {
            let start = data_start + read_u32(&data, offsets_start + index * 4)? as usize;
            let end = data_start + read_u32(&data, offsets_start + (index + 1) * 4)? as usize;
            let record = data
                .get(start..end)
                .ok_or_else(|| anyhow::anyhow!("truncated punctuation maplist record"))?;
            let separator = record
                .iter()
                .position(|byte| *byte == b'\t')
                .ok_or_else(|| anyhow::anyhow!("invalid punctuation maplist record"))?;
            let source = record[..separator].to_vec();
            let pieces = std::str::from_utf8(&record[separator + 1..])?
                .split_whitespace()
                .map(str::to_string)
                .collect();
            entries.push((source, pieces));
        }
        Ok(Self { entries })
    }

    fn lookup(&self, token: &str) -> Option<Vec<String>> {
        let target = token.as_bytes();
        self.entries
            .binary_search_by(|(source, _)| source.as_slice().cmp(target))
            .ok()
            .map(|index| self.entries[index].1.clone())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PunctuationModel {
    weights: WeightArchive,
    token_to_id: HashMap<String, usize>,
    maplist: PunctuationMap,
}

impl PunctuationModel {
    pub(crate) const MAX_LENGTH: usize = 512;
    const HIDDEN_SIZE: usize = 312;
    const HEAD_COUNT: usize = 12;
    const HEAD_WIDTH: usize = 26;
    const LAYER_COUNT: usize = 4;

    pub(crate) fn load(weights: &Path, vocabulary: &Path, maplist: &Path) -> Result<Self> {
        let weights = WeightArchive::load(weights, true)?;
        let vocabulary = std::fs::read_to_string(vocabulary)?
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let token_to_id = vocabulary
            .iter()
            .enumerate()
            .map(|(index, token)| (token.clone(), index))
            .collect::<HashMap<_, _>>();
        for token in ["[CLS]", "[SEP]", "[UNK]"] {
            if !token_to_id.contains_key(token) {
                bail!("punctuation vocabulary does not contain {}", token)
            }
        }
        Ok(Self {
            weights,
            token_to_id,
            maplist: PunctuationMap::load(maplist)?,
        })
    }

    pub(crate) fn wordpiece(&self, token: &str) -> Vec<String> {
        if let Some(mapped) = self.maplist.lookup(token) {
            return mapped;
        }
        let mut pieces = Vec::new();
        let mut buffered = String::new();
        let flush = |buffered: &mut String, pieces: &mut Vec<String>| {
            if buffered.is_empty() {
                return;
            }
            if let Some(mapped) = self.maplist.lookup(buffered) {
                pieces.extend(mapped);
            } else {
                pieces.extend(self.greedy_wordpiece(buffered));
            }
            buffered.clear();
        };
        for character in token.chars() {
            if is_cjk(character) {
                flush(&mut buffered, &mut pieces);
                let value = character.to_string();
                if let Some(mapped) = self.maplist.lookup(&value) {
                    pieces.extend(mapped);
                } else {
                    pieces.extend(self.greedy_wordpiece(&value));
                }
            } else if character.is_whitespace() {
                flush(&mut buffered, &mut pieces);
            } else {
                buffered.push(character);
            }
        }
        flush(&mut buffered, &mut pieces);
        if pieces.is_empty() {
            vec!["[UNK]".to_string()]
        } else {
            pieces
        }
    }

    pub(crate) fn predict(&self, tokens: &[String], final_input: bool) -> Result<Vec<usize>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut token_ids = vec![self.token_to_id["[CLS]"]];
        let mut final_positions = Vec::with_capacity(tokens.len());
        for token in tokens {
            for piece in self.wordpiece(token) {
                token_ids.push(
                    self.token_to_id
                        .get(&piece)
                        .copied()
                        .unwrap_or(self.token_to_id["[UNK]"]),
                );
            }
            final_positions.push(token_ids.len() - 1);
        }
        token_ids.push(self.token_to_id["[SEP]"]);
        let length = token_ids.len();
        if length > Self::MAX_LENGTH {
            bail!("punctuation model input exceeds 512 wordpieces")
        }
        let (_, embeddings) = self.weights.f32("encoder_embed_words_weight")?;
        let mut hidden = vec![0.0; length * Self::HIDDEN_SIZE];
        for (row, token_id) in token_ids.iter().copied().enumerate() {
            let source = token_id * Self::HIDDEN_SIZE;
            hidden[row * Self::HIDDEN_SIZE..(row + 1) * Self::HIDDEN_SIZE]
                .copy_from_slice(&embeddings[source..source + Self::HIDDEN_SIZE]);
        }
        let (_, types) = self.weights.f32("encoder_embed_types_out_weight")?;
        let (_, positions) = self.weights.f32("encoder_position_embeddings")?;
        for row in 0..length {
            for channel in 0..Self::HIDDEN_SIZE {
                hidden[row * Self::HIDDEN_SIZE + channel] +=
                    types[channel * 2] + positions[row * Self::HIDDEN_SIZE + channel];
            }
        }
        hidden = layer_norm(
            &hidden,
            length,
            Self::HIDDEN_SIZE,
            self.weights.f32("encoder_embed_layer_norm_beta")?.1,
            self.weights.f32("encoder_embed_layer_norm_gamma")?.1,
            1.0e-12,
        )?;
        for row in 0..length {
            hidden[row * Self::HIDDEN_SIZE + Self::HIDDEN_SIZE - 1] =
                f32::from(final_input && row + 2 == length);
        }

        for layer in 0..Self::LAYER_COUNT {
            let prefix = format!("encoder_layers_{}", layer);
            let qkv = self.linear_quantized(
                &hidden,
                length,
                Self::HIDDEN_SIZE,
                &(prefix.clone() + "_self_attn_in_proj"),
            )?;
            let mut query = vec![0.0; Self::HEAD_COUNT * length * Self::HEAD_WIDTH];
            let mut key = vec![0.0; query.len()];
            let mut value = vec![0.0; query.len()];
            let scale = 1.0_f64 / (Self::HEAD_WIDTH as f64).sqrt();
            for row in 0..length {
                for head in 0..Self::HEAD_COUNT {
                    for channel in 0..Self::HEAD_WIDTH {
                        let destination =
                            (head * length + row) * Self::HEAD_WIDTH + channel;
                        let source = row * Self::HIDDEN_SIZE * 3
                            + head * Self::HEAD_WIDTH
                            + channel;
                        query[destination] = (qkv[source] as f64 * scale) as f32;
                        key[destination] = qkv[source + Self::HIDDEN_SIZE];
                        value[destination] = qkv[source + Self::HIDDEN_SIZE * 2];
                    }
                }
            }
            let scores = punctuation_qk(
                &query,
                &key,
                Self::HEAD_COUNT,
                length,
                Self::HEAD_WIDTH,
            )?;
            let probabilities = punctuation_softmax(
                &scores,
                Self::HEAD_COUNT * length,
                length,
            )?;
            let context = punctuation_context(
                &probabilities,
                &value,
                Self::HEAD_COUNT,
                length,
                length,
                Self::HEAD_WIDTH,
            )?;
            let mut merged = vec![0.0; length * Self::HIDDEN_SIZE];
            for row in 0..length {
                for head in 0..Self::HEAD_COUNT {
                    for channel in 0..Self::HEAD_WIDTH {
                        merged[row * Self::HIDDEN_SIZE + head * Self::HEAD_WIDTH + channel] =
                            context[(head * length + row) * Self::HEAD_WIDTH + channel];
                    }
                }
            }
            let attention_output = self.linear_quantized(
                &merged,
                length,
                Self::HIDDEN_SIZE,
                &(prefix.clone() + "_self_attn_out_proj"),
            )?;
            add_in_place(&mut hidden, &attention_output)?;
            hidden = layer_norm(
                &hidden,
                length,
                Self::HIDDEN_SIZE,
                self.weights.f32(&(prefix.clone() + "_layer_norms_0_beta"))?.1,
                self.weights.f32(&(prefix.clone() + "_layer_norms_0_gamma"))?.1,
                1.0e-6,
            )?;
            let mut feed_forward = self.linear_quantized(
                &hidden,
                length,
                Self::HIDDEN_SIZE,
                &(prefix.clone() + "_fc1"),
            )?;
            for value in &mut feed_forward {
                *value = 0.5
                    * *value
                    * (1.0 + libm::erff(*value / 2.0_f32.sqrt()));
            }
            feed_forward = self.linear_quantized(
                &feed_forward,
                length,
                feed_forward.len() / length,
                &(prefix.clone() + "_fc2"),
            )?;
            add_in_place(&mut hidden, &feed_forward)?;
            hidden = layer_norm(
                &hidden,
                length,
                Self::HIDDEN_SIZE,
                self.weights.f32(&(prefix.clone() + "_layer_norms_1_beta"))?.1,
                self.weights.f32(&(prefix + "_layer_norms_1_gamma"))?.1,
                1.0e-6,
            )?;
        }
        let logits = self.linear_quantized(
            &hidden,
            length,
            Self::HIDDEN_SIZE,
            "encoder_classfier_out",
        )?;
        let scores = punctuation_softmax(&logits, length, 5)?;
        Ok(final_positions
            .into_iter()
            .map(|position| {
                let row = &scores[position * 5..position * 5 + 5];
                let mut label = argmax(row);
                if label == 1 && f64::from(row[0]) / f64::from(row[1]) > 0.6 {
                    label = 0;
                }
                label
            })
            .collect())
    }

    fn greedy_wordpiece(&self, token: &str) -> Vec<String> {
        if self.token_to_id.contains_key(token) {
            return vec![token.to_string()];
        }
        let characters = token.chars().collect::<Vec<_>>();
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < characters.len() {
            let mut found = None;
            for end in (start + 1..=characters.len()).rev() {
                let mut candidate = characters[start..end].iter().collect::<String>();
                if start > 0 {
                    candidate.insert_str(0, "@@");
                }
                if self.token_to_id.contains_key(&candidate) {
                    found = Some((candidate, end));
                    break;
                }
            }
            let Some((piece, end)) = found else {
                return vec!["[UNK]".to_string()];
            };
            pieces.push(piece);
            start = end;
        }
        pieces
    }

    fn linear_quantized(
        &self,
        input: &[f32],
        rows: usize,
        inner: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let (_, scale) = self.weights.f32(&(prefix.to_string() + "_data_scale"))?;
        let quantized = input
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let scale = broadcast_value(scale, index, inner);
                let scaled = *value / scale;
                let rounded = if scaled > 0.0 { scaled + 0.5 } else { scaled - 0.5 };
                rounded.trunc().clamp(-128.0, 127.0) as i8
            })
            .collect::<Vec<_>>();
        let (weight_shape, weight) = self.weights.i8(&(prefix.to_string() + "_weight"))?;
        let columns = weight_shape.first().copied().ok_or_else(|| {
            anyhow::anyhow!("punctuation weight {} has no output dimension", prefix)
        })?;
        let accumulator = punctuation_quantized_linear(
            &quantized,
            weight,
            rows,
            columns,
            inner,
        )?;
        let (_, requant) = self
            .weights
            .f32(&(prefix.to_string() + "_data_requant_scale"))?;
        let bias = if self.weights.contains(&(prefix.to_string() + "_bias")) {
            Some(self.weights.f32(&(prefix.to_string() + "_bias"))?.1)
        } else {
            None
        };
        Ok(accumulator
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let column = index % columns;
                value as f32 * broadcast_value(requant, column, columns)
                    + bias.map_or(0.0, |values| broadcast_value(values, column, columns))
            })
            .collect())
    }
}

fn linear(
    input: &[f32],
    rows: usize,
    inner: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    let columns = bias.len();
    if input.len() != rows * inner || weight.len() != columns * inner {
        bail!("postprocess linear shape mismatch")
    }
    let mut output = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut value = bias[column];
            for index in 0..inner {
                value += input[row * inner + index] * weight[column * inner + index];
            }
            output[row * columns + column] = value;
        }
    }
    Ok(output)
}

fn layer_norm(
    input: &[f32],
    rows: usize,
    width: usize,
    beta: &[f32],
    gamma: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    if input.len() != rows * width || beta.len() != width || gamma.len() != width {
        bail!("postprocess layer norm shape mismatch")
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let values = &input[row * width..(row + 1) * width];
        let mean = values.iter().sum::<f32>() / width as f32;
        let variance = values
            .iter()
            .map(|value| (*value - mean) * (*value - mean))
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for column in 0..width {
            output[row * width + column] =
                beta[column] + gamma[column] * (values[column] - mean) * inverse;
        }
    }
    Ok(output)
}

fn softmax_rows(input: &[f32], rows: usize, columns: usize) -> Result<Vec<f32>> {
    punctuation_softmax(input, rows, columns)
}

fn add_in_place(left: &mut [f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        bail!("postprocess residual shape mismatch")
    }
    for (left, right) in left.iter_mut().zip(right) {
        *left += *right;
    }
    Ok(())
}

fn add_positional_encoding(values: &mut [f32], rows: usize, width: usize) {
    let half = width / 2;
    let scale = 10_000.0_f32.ln() / (half - 1) as f32;
    for row in 0..rows {
        for index in 0..half {
            let frequency = (-(index as f32) * scale).exp();
            values[row * width + index] += (row as f32 * frequency).sin();
            values[row * width + half + index] += (row as f32 * frequency).cos();
        }
    }
}

fn quantize_i16(values: &[f32]) -> Vec<i16> {
    values
        .iter()
        .map(|value| {
            let scaled = if *value > 0.0 {
                *value * 1024.0 + 0.5
            } else {
                *value * 1024.0 - 0.5
            };
            scaled.trunc().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

fn native_tanh(value: f32) -> f32 {
    (2.0_f64 / (1.0 + (-2.0_f64 * value as f64).exp()) - 1.0) as f32
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn broadcast_value(values: &[f32], index: usize, width: usize) -> f32 {
    match values.len() {
        0 => 0.0,
        1 => values[0],
        length if length == width => values[index % width],
        length => values[index % length],
    }
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow::anyhow!("truncated punctuation maplist"))?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2F800..=0x2FA1F
    )
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::time::{SystemTime, UNIX_EPOCH};

    use ndarray::{ArrayD, IxDyn};
    use ndarray_npy::NpzWriter;

    use super::{WeightArchive, layer_norm};

    #[test]
    fn weight_archive_loads_mixed_f32_and_i8_arrays() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "iflytek-postprocess-{}-{}.npz",
            std::process::id(),
            nonce
        ));
        let mut writer = NpzWriter::new(File::create(&path).unwrap());
        let floats = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 2.0]).unwrap();
        let integers = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1_i8, 2]).unwrap();
        writer.add_array("float", &floats).unwrap();
        writer.add_array("integer", &integers).unwrap();
        writer.finish().unwrap();

        let archive = WeightArchive::load(&path, false).unwrap();
        assert_eq!(archive.f32("float").unwrap().1, [1.0, 2.0]);
        assert_eq!(archive.i8("integer").unwrap().1, [-1, 2]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn layer_norm_applies_beta_then_gamma() {
        let output = layer_norm(&[1.0, 3.0], 1, 2, &[10.0, 20.0], &[2.0, 4.0], 0.0)
            .unwrap();
        assert_eq!(output, [8.0, 24.0]);
    }
}
