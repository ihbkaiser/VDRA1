# Báo cáo cuối: chẩn đoán rebuttal VDRA

## Trạng thái xuất bản và kiểm tra

Toàn bộ công việc đã commit trên nhánh cục bộ `rebuttal`; kết quả iter150 nằm trong commit `0cd2909`, báo cáo này được commit sau đó. Bảy lần push tới `origin/rebuttal` đều bị GitHub từ chối HTTP 403 (`Permission to ihbkaiser/VDRA1.git denied to ihbkaiser`), nên không có commit URL từ xa. Lần thử gần nhất là 2026-10-31 09:16 UTC. Việc publish đang bị chặn cho tới khi tài khoản có quyền ghi vào repository đích; không lên lịch thử lại tự động. `/workspace` không phải volume bền vững trên instance này.

`gh_token` và `hf_token` đang bị ignore, untracked và không có trong lịch sử commit. Bộ kiểm thử CPU cuối cùng: `pytest -q rebuttal_igpo/tests` → 41 passed.

## §2 — mã nguồn và provenance cho từng run

Không có archive nào chứa commit đào tạo, resolved config hoặc log đủ để xác định chính xác revision đã tạo checkpoint đó. So sánh nguồn tìm thấy không có ref IGPO nào khớp hoàn toàn với snapshot Gear.

| Run trong archive | Commit đào tạo xác nhận được? | Kết luận đối chiếu |
|---|---|---|
| VDRA iteration 50 | Không | Chỉ có checkpoint. Snapshot Gear là ứng viên để viết probe, không xác nhận đây là mã đã train run. |
| VDRA iteration 100 | Không | Tương tự; không có run metadata hoặc cây rollout để tái dựng cấu hình. |
| VDRA iteration 150 | Không | Tương tự; phép đo X là frozen-checkpoint probe. |
| VDRA iteration 235 | Không | Tương tự; archive có checkpoint và tokenizer nhưng không có commit/config đào tạo. |
| SPO iteration 235 | Không | Archive chỉ có checkpoint; tokenizer và run metadata đều thiếu. |

Trong các đường dẫn chung với IGPO, `origin/main` và `origin/fix/budget-allocation-theory` cùng có 592 file khớp và 69 file khác. Commit gần nhất theo ngày là `94fe2a6` trên `origin/codex/locate-evaluate.sh-code`, nhưng vẫn khác 70 file chung. Bộ cấp phát Gear, estimator TV và hàng đợi online chỉ có trong snapshot. Vì thiếu provenance, các phân tích mang nhãn “GEAR snapshot defaults” là đối chứng mã nguồn, không phải replay as-trained. Chi tiết nằm trong `CODE_SYNC_REPORT.md`, `RUN_PROVENANCE.md` và `MECHANISM_AUDIT.md`.

## Ba thí nghiệm ưu tiên trên RTX 3090 đã chạy

1. **A3/A2/A4/A5 tại iteration 100:** 112 parent states trên 16 prompt clusters, 12,029,177 continuation tokens. Spearman giữa proxy variance và variance của `R_b` là −0.148, CI bootstrap theo prompt [−0.518, 0.259]. Noise lớn làm tăng allocation churn; A4 tương quan TV/value yếu; các so sánh A5 là MC proxies.
2. **A7 tại iteration 235:** sáu lần đánh giá theo cặp trên AIME24/AIME25/AMC23, 16 samples/question, tổng 797 s. Chỉ AIME24 Pass@16 có CI unadjusted không cắt 0; kết quả là single-seed diagnostic, không phải benchmark chính thức. SPO không có tokenizer nên tokenizer VDRA dùng chung và identity chưa xác minh.
3. **X tại iterations 50 và 150:** mỗi bank có 112/112 parent states và đã chạy A2/A3/A4/A5. Iter50: 13,327,123 continuation tokens, 5,853.9 s, rho A3 0.175 [−0.190, 0.440]. Iter150: 11,205,934 tokens, 5,460.7 s, rho A3 −0.342 [−0.653, 0.062]. Raw banks ở ngoài Git; compact CSV/JSON, hash và diễn giải ở `results/x_iter050/`, `results/x_iter150/` và `results/MANIFEST.md`.

## Lệnh đầu tiên để tái chạy X

Từ repo đã chuẩn bị trên instance:

```bash
cd /workspace/VDRA1/.worktrees/rebuttal
/workspace/rebuttal_out/run_x_probe.sh 050
```

Để chạy iteration 150 sau khi iter050 và vLLM đã dừng sạch, dùng `/workspace/rebuttal_out/run_x_probe.sh 150`. Script dùng loopback, tokenizer đã cố định, alias theo checkpoint, gzip có thể resume và cấu hình 112 parent đã preregister.

## Dữ liệu cần chuyển tới server để un-block A1/A6

Các checkpoint và dữ liệu MATH cần thiết cho probe đã có trên instance. Để làm replay trung thực của training và reconcile các bảng/figures của paper, cần đưa đúng artifacts của từng run lên máy phân tích:

- run config đã resolve, `metadata.json`/dotenv, commit SHA và log đầu/cuối;
- stored rollout trees (`traj_results_ds`), demos và queue/allocation records;
- `merged_edges`, `episodes/merged`, `training_timing.jsonl` và evaluation outputs;
- paper/supplement hoặc dữ liệu nguồn cho các bảng và hình cần đối chiếu.

Không có những file đó thì A1 và A6 vẫn blocked; không thể suy ra chúng từ checkpoint weights hoặc log W&B không liên quan. Hãy giữ raw banks/generations ngoài repository; chỉ chuyển chúng qua kho nội bộ được phép nếu cần lưu lâu dài.

## Có thể hoàn tất trước hạn và phần để camera-ready

Tất cả hạng mục khả thi trên instance đã hoàn tất trước cutoff sinh GPU 2026-11-01 01:00 UTC: A0 inventory/code audit, T derivation/simulation, S smoke, A3/A2/A4/A5 ở iter100/50/150, A7, tests, summaries và figures. A1/A6 không thể hoàn tất nếu không có artifacts nêu trên. Trước cutoff còn lại chỉ có status/manifest, báo cáo và thử push ở chu kỳ kế tiếp; hai bank X đã kết thúc lúc 03:13 UTC ngày 2026-10-31.

Để camera-ready, ưu tiên mở rộng prompt clusters và seeds, kiểm tra sensitivity của estimator ở nhiều MC budgets, đánh giá A7 ở context dài hơn nếu đúng với protocol của paper, và đo trực tiếp sai số Eq. (5)/clipped-PPO gradient trên saved training edges. Không diễn giải các `J_Rb`, `L_node`, `L_edge`, `L_heldout` hiện tại như ground truth gradient error.
