#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include "slide-leaf-dragon/binary/module/prerank/ensemble/prerank_ranker.h"
#include "ks/algo-engine/formula1/src/formula1.h"
#include "dragon/src/util/id_mapping_util.h"
namespace ks {
namespace reco {
namespace slide_leaf {
namespace prerank {
class CommentCopyForwardScore : public ScoreCalculator {
 public:
  ks::reco::formula1::Formula1 f1_;
  folly::F14FastMap<uint64, double> pid2score;
  CommentCopyForwardScore():
    ScoreCalculator("CommentCopyForwardScore"),
    f1_("formula.scenarioKey72.slide_prerank_CommentCopyForwardScore_f1")  // rename later
  {}
  void Init(SlideContext *slide_context, const ks::reco::UserInfo &user_info) override {
    if (slide_context == nullptr) {
      return;
    }
    auto &session = slide_context->session;
    enable_multiply_ = session.prerank_enable_comment_copy_forward_score();
    multiply_params_ = session.prerank_comment_copy_forward_score_params();
    // f1
    pid2score.clear();
    enable_prerank_get_f1_comment_copy_forward_score_ =
        session.enable_prerank_get_f1_comment_copy_forward_score();
  }

  bool IsEnabledMultiply(SlideContext *slide_context,
                 const ks::reco::UserInfo &user_info) const override {
    return enable_multiply_;
  }

  std::string GetMultiplyParams(SlideContext *slide_context,
                                const ks::reco::UserInfo &user_info) const override {
    return multiply_params_;  // alpha beta bias norm
  }
  void PreCalc(SlideContext *slide_context, const ks::reco::UserInfo &user_info,
               const std::vector<PrerankScoreInfo *> &results) {
    if (enable_prerank_get_f1_comment_copy_forward_score_) {
      std::vector<std::vector<double>> pxtrs(8);
      std::vector<std::string> pxtr_names = {
        "pctr", "pvtr", "plvtr", "pltr", "pwtr", "pftr", "pcmtr", "pcommentforwardcopy"
      };

      if (pxtr_names.size() != pxtrs.size()) return;
      for (const auto &score_info : results) {
        if (score_info == nullptr || score_info->pr_result == nullptr) continue;
        double pctr = score_info->pr_result->pctr;
        double pvtr = score_info->pr_result->pvtr;
        double plvtr = score_info->pr_result->plvtr;
        double pltr = score_info->pr_result->pltr;
        double pwtr = score_info->pr_result->pwtr;
        double pftr = score_info->pr_result->pftr;
        double pcmtr = score_info->pr_result->pcmtr;
        double pcommentforwardcopy = score_info->pr_result->pcommentforwardcopy;

        pxtrs[0].push_back(std::max(pctr, 0.0));
        pxtrs[1].push_back(std::max(pvtr, 0.0));
        pxtrs[2].push_back(std::max(plvtr, 0.0));
        pxtrs[3].push_back(std::max(pltr, 0.0));
        pxtrs[4].push_back(std::max(pwtr, 0.0));
        pxtrs[5].push_back(std::max(pftr, 0.0));
        pxtrs[6].push_back(std::max(pcmtr, 0.0));
        pxtrs[7].push_back(std::max(pcommentforwardcopy, 0.0));
      }
      f1_.Reset();
      f1_.SetCommonData("is_gamora", user_info.is_gamora());
      f1_.SetCommonData("is_nebula", user_info.is_nebula_user());
      for (int i = 0; i < pxtrs.size(); i++) {
        f1_.SetItemData(pxtr_names[i], pxtrs[i]);
      }
      auto ab_mapping_id = ::ks::platform::GetABMappingIdFromRequest(slide_context->context->GetRequest());
      if (!f1_.Calc(ks::AbtestBiz::THANOS_RECO, user_info.id(), user_info.device_id(), false,
                    ab_mapping_id)) {
        LOG_EVERY_N(INFO, 10000) << "Prerank CommentCopyForwardScore: f1 cal failed";
        return;
      }
      const auto f1_score_out = f1_.GetValue("score");
      if (!f1_score_out) {
        LOG_EVERY_N(INFO, 10000) << "Prerank CommentCopyForwardScore: f1 output not found node of 'score'";
        return;
      }

      if (f1_score_out->size() != results.size()) {
        LOG_EVERY_N(INFO, 10000) << "Prerank CommentCopyForwardScore: f1.output.size != prerank_result.size"
                  << ", f1.size=" << f1_score_out->size() << ", result.size=" << results.size();
        return;
      }

      for (int i = 0; i < results.size() && i < f1_score_out->size(); i++) {
        if (results[i] == nullptr) continue;
        uint64 pid = results[i]->photo_id;
        double score = f1_score_out->at(i);
        LOG_EVERY_N(INFO, 10000) << "Prerank CommentCopyForwardScore: pid=" << pid << ", score=" << score;
        pid2score[pid] = score;
      }
    }
  }
  double GetScore(SlideContext *slide_context, const ks::reco::UserInfo &user_info,
                  const PrerankScoreInfo &score_info) const override {
    if (score_info.pr_result == nullptr) return 0.0;
    uint64 pid = score_info.photo_id;
    if (enable_prerank_get_f1_comment_copy_forward_score_) {
      if (pid2score.find(pid) != pid2score.end()) {
        return pid2score.at(pid);
      }
      return 0.0;
    }
    return score_info.pr_result->pcommentforwardcopy;
  }

 private:
  bool enable_prerank_get_f1_comment_copy_forward_score_ = false;
};

}  // namespace prerank
}  // namespace slide_leaf
}  // namespace reco
}  // namespace ks
