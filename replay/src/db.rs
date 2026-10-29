//! A dimos memory2 recording (.db): every selected stream merged into one log-time order, a
//! page of ids at a time so a recording of many gigabytes is never indexed in memory.

use std::collections::VecDeque;
use std::path::Path;

use rusqlite::{Connection, OpenFlags};

const PAGE: i64 = 4096;

pub struct Record {
    pub stream: usize,
    /// Log time, seconds.
    pub ts: f64,
    pub data: Vec<u8>,
}

struct Cursor {
    name: String,
    page: VecDeque<(i64, f64)>,
    last_ts: f64,
    done: bool,
}

pub struct Recording {
    connection: Connection,
    cursors: Vec<Cursor>,
    end_ts: f64,
}

fn quote(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

impl Recording {
    /// `streams` in the order records report them; each must use the lcm codec. The window is
    /// seconds from the recording's first message; a duration of 0 runs to the end.
    pub fn open(
        path: &Path,
        streams: &[String],
        start_s: f64,
        duration_s: f64,
    ) -> Result<Self, String> {
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let mut first = f64::INFINITY;
        for name in streams {
            let codec: String = connection
                .query_row(
                    "SELECT json_extract(config, '$.codec_id') FROM _streams WHERE name = ?1",
                    [name],
                    |row| row.get(0),
                )
                .map_err(|_| format!("{}: no stream {name:?}", path.display()))?;
            // dimos's jpeg codec is still an LCM Image, its data a JPEG; lcm::image unpacks it.
            if codec != "lcm" && codec != "jpeg" {
                return Err(format!(
                    "stream {name:?} is {codec:?}-encoded; only lcm is decoded"
                ));
            }
            let ts: Option<f64> = connection
                .query_row(&format!("SELECT MIN(ts) FROM {}", quote(name)), [], |row| {
                    row.get(0)
                })
                .map_err(|error| error.to_string())?;
            first = first.min(ts.unwrap_or(f64::INFINITY));
        }
        let start = first + start_s;
        // Cursors read strictly after last_ts, so a record exactly at the start is kept.
        let cursors = streams
            .iter()
            .map(|name| Cursor {
                name: name.clone(),
                page: VecDeque::new(),
                last_ts: start - 1e-9,
                done: false,
            })
            .collect();
        let end_ts = if duration_s > 0.0 {
            start + duration_s
        } else {
            f64::INFINITY
        };
        Ok(Self {
            connection,
            cursors,
            end_ts,
        })
    }

    fn fill(&mut self, index: usize) -> Result<(), String> {
        let cursor = &mut self.cursors[index];
        let sql = format!(
            "SELECT id, ts FROM {} WHERE ts > ?1 AND ts <= ?2 ORDER BY ts LIMIT ?3",
            quote(&cursor.name)
        );
        let mut statement = self
            .connection
            .prepare_cached(&sql)
            .map_err(|e| e.to_string())?;
        let rows = statement
            .query_map(
                rusqlite::params![cursor.last_ts, self.end_ts, PAGE],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|e| e.to_string())?
            .collect::<rusqlite::Result<Vec<(i64, f64)>>>()
            .map_err(|e| e.to_string())?;
        match rows.last() {
            None => cursor.done = true,
            Some(last) => {
                cursor.last_ts = last.1;
                cursor.page.extend(rows);
            }
        }
        Ok(())
    }

    pub fn next_record(&mut self) -> Result<Option<Record>, String> {
        for index in 0..self.cursors.len() {
            if self.cursors[index].page.is_empty() && !self.cursors[index].done {
                self.fill(index)?;
            }
        }
        let Some(stream) = (0..self.cursors.len())
            .filter(|index| !self.cursors[*index].page.is_empty())
            .min_by(|a, b| {
                self.cursors[*a].page[0]
                    .1
                    .total_cmp(&self.cursors[*b].page[0].1)
            })
        else {
            return Ok(None);
        };
        let (id, ts) = self.cursors[stream]
            .page
            .pop_front()
            .expect("checked above");
        let sql = format!(
            "SELECT data FROM {} WHERE id = ?1",
            quote(&format!("{}_blob", self.cursors[stream].name))
        );
        let mut statement = self
            .connection
            .prepare_cached(&sql)
            .map_err(|e| e.to_string())?;
        let data = statement
            .query_row([id], |row| row.get(0))
            .map_err(|e| e.to_string())?;
        Ok(Some(Record { stream, ts, data }))
    }
}
