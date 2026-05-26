# 01_preprocess.R
# Run from project root: C:\Users\joels\Documents\Claude_files\
# Cleans session conversation data and baseline survey data,
# computes survey composites, and merges datasets.

library(tidyverse)

# ── Session data ─────────────────────────────────────────────────────────────

sessions_raw <- read_csv(
  "data_files/metacog_sessions_pilotdata.csv",
  show_col_types = FALSE
)

sessions_clean <- sessions_raw %>%
  select(
    session_id, pid, topic, session_num, role, content, turn, timestamp,
    starts_with("Q20"), starts_with("Q29"), starts_with("Q30")
  ) %>%
  filter(!is.na(pid), pid != "") %>%
  mutate(
    timestamp  = as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC"),
    turn       = as.integer(turn),
    session_num = as.integer(session_num),
    across(starts_with("Q20"), as.numeric),
    across(starts_with("Q29"), as.numeric),
    across(starts_with("Q30"), as.numeric)
  )

# Survey scores appear only on the row where Q20.1 is not NA
# (the last turn of each session)
survey_session <- sessions_clean %>%
  filter(!is.na(.data[["Q20.1_AI_Responsiveness"]])) %>%
  select(pid, session_id, session_num, topic,
         starts_with("Q20"), starts_with("Q29"), starts_with("Q30")) %>%
  mutate(
    AI_attitude_composite  = rowMeans(pick(starts_with("Q20")), na.rm = TRUE),
    MetaCog_composite      = rowMeans(pick(starts_with("Q29")), na.rm = TRUE),
    InfoConfidence         = Q30_1_InfoConfidence
  )

write_csv(sessions_clean,   "output/processed/sessions_clean.csv")
write_csv(survey_session,   "output/processed/survey_session_scores.csv")
message("Session data saved.")

# ── Baseline data ─────────────────────────────────────────────────────────────
# Row 1 = compact column names (used as header by read_csv).
# Row 2 = full Qualtrics question-text (multi-line quoted strings) → drop it.

baseline_raw <- read_csv(
  "data_files/metacog_ai_baseline_pilotdata.csv",
  show_col_types = FALSE
)
baseline <- baseline_raw[-1, ]   # drop Qualtrics question-text row

# Q25: 12-item matrix, 1-7 scale. Values outside [1,7] are test/invalid → NA.
q25_cols <- paste0("Q25_", 1:12)
baseline <- baseline %>%
  mutate(across(all_of(q25_cols), ~ {
    x <- as.numeric(.)
    ifelse(x >= 1 & x <= 7, x, NA_real_)
  }))

# Q21-Q23: LLM attitude scales, 1-7
likert_cols <- c(paste0("Q21_", 1:4), paste0("Q22_", 1:6), paste0("Q23_", 1:3))
baseline <- baseline %>%
  mutate(across(all_of(likert_cols), as.numeric))

# MCQ-30: metacognitive beliefs questionnaire (30 items, 1-4 scale)
mcq_cols <- paste0("mcq_30_", 1:30)
baseline <- baseline %>%
  mutate(across(all_of(mcq_cols), as.numeric))

# MCQ-30 subscale scoring (Wells & Cartwright-Hatton, 2004)
baseline <- baseline %>%
  mutate(
    MCQ_positive_worry      = rowMeans(pick(mcq_30_1,  mcq_30_7,  mcq_30_10, mcq_30_19, mcq_30_23, mcq_30_28), na.rm = TRUE),
    MCQ_uncontrollability   = rowMeans(pick(mcq_30_2,  mcq_30_4,  mcq_30_9,  mcq_30_11, mcq_30_15, mcq_30_21), na.rm = TRUE),
    MCQ_cognitive_confidence= rowMeans(pick(mcq_30_8,  mcq_30_14, mcq_30_17, mcq_30_24, mcq_30_26, mcq_30_29), na.rm = TRUE),
    MCQ_need_control        = rowMeans(pick(mcq_30_6,  mcq_30_13, mcq_30_20, mcq_30_22, mcq_30_25, mcq_30_27), na.rm = TRUE),
    MCQ_self_consciousness  = rowMeans(pick(mcq_30_3,  mcq_30_5,  mcq_30_12, mcq_30_16, mcq_30_18, mcq_30_30), na.rm = TRUE),
    MCQ_total               = rowMeans(pick(all_of(mcq_cols)), na.rm = TRUE),
    LLM_affect              = rowMeans(pick(Q21_1, Q21_2, Q21_3, Q21_4), na.rm = TRUE),
    LLM_credibility         = rowMeans(pick(Q22_1, Q22_2, Q22_3, Q22_4, Q22_5, Q22_6), na.rm = TRUE),
    LLM_intelligence        = rowMeans(pick(Q23_1, Q23_2, Q23_3), na.rm = TRUE),
    ChatGPT_quality         = rowMeans(pick(all_of(q25_cols)), na.rm = TRUE)
  ) %>%
  rename(pid = ResponseId)

write_csv(baseline, "output/processed/baseline_clean.csv")
message("Baseline data saved.")

# ── Merge session survey with baseline ────────────────────────────────────────

baseline_scores <- baseline %>%
  select(pid, topic,
         MCQ_positive_worry, MCQ_uncontrollability, MCQ_cognitive_confidence,
         MCQ_need_control, MCQ_self_consciousness, MCQ_total,
         LLM_affect, LLM_credibility, LLM_intelligence, ChatGPT_quality)

survey_merged <- survey_session %>%
  left_join(baseline_scores, by = "pid", suffix = c("_session", "_baseline"))

n_matched <- sum(!is.na(survey_merged$MCQ_total))
message(sprintf("Merged: %d / %d session participants matched to baseline.",
                n_matched, nrow(survey_merged)))

write_csv(survey_merged, "output/processed/survey_merged.csv")
message("Preprocessing complete. All files in output/processed/")
