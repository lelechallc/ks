#pragma once
#include <string>
#include <algorithm>
#include <vector>
#include "slide-leaf-dragon/binary/module/prerank/ensemble/prerank_ranker.h"
#include "slide-leaf-dragon/binary/util/common_util.h"
#include "dragon/src/util/id_mapping_util.h"
namespace ks {
namespace reco {
namespace slide_leaf {
namespace prerank {
class PxtrRelativeQuantileScore : public ScoreCalculator {
 private:
  bool enable_quantile_cal_exp_gamora_ = false;
  bool enable_quantile_cal_sigmoid_gamora_ = false;
  std::vector<double> param_alpha_vec_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> param_k_vec_gamora_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> pxtrs_max_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> pxtrs_min_gamora_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> pxtrs_quan_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> param_quan_center_gamora_ = {80.0, 80.0, 80.0, 80.0,
                                      80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0};
  std::vector<double> param_sigmoid_a_vec_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> param_sigmoid_b_vec_gamora_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

 public:
  PRERANK_SCORE_QUEUE_CTOR(PxtrRelativeQuantileScore)
  void Init(SlideContext *slide_context, const ks::reco::UserInfo &user_info) override {
    auto &session = slide_context->session;
    enable_multiply_ = session.prerank_enable_pxtr_relative_quantile_score();
    multiply_params_ = session.prerank_pxtr_relative_quantile_score_multiply_params();
    enable_quantile_cal_exp_gamora_ = session.prerank_pxtr_relative_quantile_score_enable_cal_exp();
    enable_quantile_cal_sigmoid_gamora_ = session.prerank_pxtr_relative_quantile_score_enable_cal_sigmoid();
  }

  bool IsEnabledMultiply(SlideContext *slide_context,
                         const ks::reco::UserInfo &user_info) const override {
    return enable_multiply_;
  }
  void PreCalc(SlideContext *slide_context, const ks::reco::UserInfo &user_info,
               const std::vector<PrerankScoreInfo *> &results) {
    param_alpha_vec_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param_k_vec_gamora_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    param_quan_center_gamora_ = {80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0};
    pxtrs_max_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    pxtrs_min_gamora_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    pxtrs_quan_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param_sigmoid_a_vec_gamora_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param_sigmoid_b_vec_gamora_ = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    std::vector<std::vector<double>> pxtrs(11);
    std::vector<std::string> pxtr_names = {
      "pctr", "pvtr", "plvtr", "pltr", "pwtr", "pftr", "pdtr", "plstr", "plsst", "pcmtr", "psvr"
    };
    if (pxtr_names.size() != pxtrs.size() || pxtrs_max_gamora_.size() != pxtrs.size()
        || pxtrs_min_gamora_.size() != pxtrs.size() || pxtrs_quan_gamora_.size() != pxtrs.size()
        || param_quan_center_gamora_.size() != pxtrs.size()) return;
    for (const auto &score_info : results) {
      if (score_info->pr_result == nullptr) {
        pxtrs[0].push_back(0);
        pxtrs[1].push_back(0);
        pxtrs[2].push_back(0);
        pxtrs[3].push_back(0);
        pxtrs[4].push_back(0);
        pxtrs[5].push_back(0);
        pxtrs[6].push_back(0);
        pxtrs[7].push_back(0);
        pxtrs[8].push_back(0);
        pxtrs[9].push_back(0);
        pxtrs[10].push_back(0);
      } else {
        double pctr = score_info->pr_result->pctr;
        double pvtr = score_info->pr_result->pvtr;
        double plvtr = score_info->pr_result->plvtr;
        double pltr = score_info->pr_result->pltr;
        double pwtr = score_info->pr_result->pwtr;
        double pftr = score_info->pr_result->pftr;
        double pdtr = score_info->pr_result->pdtr;
        double plstr = score_info->pr_result->plstr;
        double plsst = score_info->pr_result->plsst;
        double pcmtr = score_info->pr_result->pcmtr;
        double psvtr = score_info->pr_result->psvtr;
        pxtrs[0].push_back(std::max(pctr, 0.0));
        pxtrs[1].push_back(std::max(pvtr, 0.0));
        pxtrs[2].push_back(std::max(plvtr, 0.0));
        pxtrs[3].push_back(std::max(pltr, 0.0));
        pxtrs[4].push_back(std::max(pwtr, 0.0));
        pxtrs[5].push_back(std::max(pftr, 0.0));
        pxtrs[6].push_back(std::max(pdtr, 0.0));
        pxtrs[7].push_back(std::max(plstr, 0.0));
        pxtrs[8].push_back(std::max(plsst, 0.0));
        pxtrs[9].push_back(std::max(pcmtr, 0.0));
        pxtrs[10].push_back(std::max(psvtr, 0.0));
      }
      for (int i = 0; i < pxtrs_max_gamora_.size() && i < pxtrs_min_gamora_.size(); i++) {
        pxtrs_max_gamora_[i] = std::max(pxtrs[i].back(), pxtrs_max_gamora_[i]);
        pxtrs_min_gamora_[i] = std::min(pxtrs[i].back(), pxtrs_min_gamora_[i]);
      }
    }

    auto &session = slide_context->session;
    std::string param_alpha = "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
    std::string param_k = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0";
    std::string param_quan_center = "80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0";
    std::string param_sigmoid_a = "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
    std::string param_sigmoid_b = "2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0";
    param_alpha = session.prerank_pxtr_relative_quantile_score_alpha();
    param_k = session.prerank_pxtr_relative_quantile_score_k();
    param_quan_center = session.prerank_pxtr_relative_quantile_score_quan_center();
    param_sigmoid_a = session.prerank_pxtr_relative_quantile_score_sigmoid_a();
    param_sigmoid_b = session.prerank_pxtr_relative_quantile_score_sigmoid_b();

    ks::reco::StringToDoubleVec(param_alpha, &param_alpha_vec_gamora_);
    ks::reco::StringToDoubleVec(param_k, &param_k_vec_gamora_);
    ks::reco::StringToDoubleVec(param_quan_center, &param_quan_center_gamora_);
    ks::reco::StringToDoubleVec(param_sigmoid_a, &param_sigmoid_a_vec_gamora_);
    ks::reco::StringToDoubleVec(param_sigmoid_b, &param_sigmoid_b_vec_gamora_);
    if (param_alpha_vec_gamora_.size() != pxtrs.size() || param_k_vec_gamora_.size() != pxtrs.size()) {
      LOG_EVERY_N(INFO, 100000) << "Prerank PxtrRelativeQuantileScore preCalc get param failed ";
      return;
    }
    if (enable_quantile_cal_exp_gamora_ || enable_quantile_cal_sigmoid_gamora_) {
      for (int i = 0; i < pxtrs.size() && i < param_alpha_vec_gamora_.size()
            && i < pxtrs_quan_gamora_.size(); i++) {
        if (std::abs(param_alpha_vec_gamora_[i]) > 0) {
          int p_index = (std::max(param_quan_center_gamora_[i], 0.0) * pxtrs[i].size()) / 100;
          nth_element(pxtrs[i].begin(), pxtrs[i].begin() + p_index, pxtrs[i].end());
          if (p_index < pxtrs[i].size()) {
            pxtrs_quan_gamora_[i] = pxtrs[i][p_index];
          }
        }
      }
    }
    LOG_EVERY_N(INFO, 100000) << "Prerank PxtrRelativeQuantileScore preCalcgamoraWithoutF1 ";
  }

  std::string GetMultiplyParams(SlideContext *slide_context,
                                const ks::reco::UserInfo &user_info) const override {
    return multiply_params_;  // alpha beta bias norm
  }

  double GetScore(SlideContext *slide_context, const ks::reco::UserInfo &user_info,
                  const PrerankScoreInfo &score_info) const override {
    double score_gamora = 1.0;
    if (score_info.pr_result != nullptr) {
        std::vector<double> pxtrs_gamora = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double pctr_gamora = score_info.pr_result->pctr;
        double pvtr_gamora = score_info.pr_result->pvtr;
        double plvtr_gamora = score_info.pr_result->plvtr;
        double pltr_gamora = score_info.pr_result->pltr;
        double pwtr_gamora = score_info.pr_result->pwtr;
        double pftr_gamora = score_info.pr_result->pftr;
        double pdtr_gamora = score_info.pr_result->pdtr;
        double plstr_gamora = score_info.pr_result->plstr;
        double plsst_gamora = score_info.pr_result->plsst;
        double pcmtr_gamora = score_info.pr_result->pcmtr;
        double psvtr_gamora = score_info.pr_result->psvtr;
        pxtrs_gamora[0] = std::max(pctr_gamora, 0.0);
        pxtrs_gamora[1] = std::max(pvtr_gamora, 0.0);
        pxtrs_gamora[2] = std::max(plvtr_gamora, 0.0);
        pxtrs_gamora[3] = std::max(pltr_gamora, 0.0);
        pxtrs_gamora[4] = std::max(pwtr_gamora, 0.0);
        pxtrs_gamora[5] = std::max(pftr_gamora, 0.0);
        pxtrs_gamora[6] = std::max(pdtr_gamora, 0.0);
        pxtrs_gamora[7] = std::max(plstr_gamora, 0.0);
        pxtrs_gamora[8] = std::max(plsst_gamora, 0.0);
        pxtrs_gamora[9] = std::max(pcmtr_gamora, 0.0);
        pxtrs_gamora[10] = std::max(psvtr_gamora, 0.0);

        if (pxtrs_gamora.size() != param_alpha_vec_gamora_.size()
            || pxtrs_gamora.size() != param_k_vec_gamora_.size()
            || pxtrs_gamora.size() != pxtrs_max_gamora_.size()
            || pxtrs_gamora.size() != pxtrs_min_gamora_.size()
            || pxtrs_gamora.size() != pxtrs_quan_gamora_.size()
            || pxtrs_gamora.size() != param_sigmoid_a_vec_gamora_.size()
            || pxtrs_gamora.size() != param_sigmoid_b_vec_gamora_.size()) {
              LOG_EVERY_N(INFO, 100000) <<"Prerank PxtrRelativeQuantileScore GetScore size not match"
                        << ", score= " << score_gamora;
              return 1.0;
        }
        if (enable_quantile_cal_exp_gamora_) {
          for (int i = 0; i < pxtrs_gamora.size(); i++) {
            double center = pxtrs_quan_gamora_[i];
            double relative_quantile = param_alpha_vec_gamora_[i] *
              (pxtrs_gamora[i] - center) / std::max((pxtrs_max_gamora_[i] - pxtrs_min_gamora_[i]), 1e-19);
            double relative_quantile_adjust = pow(param_k_vec_gamora_[i], relative_quantile);
            score_gamora *= relative_quantile_adjust;
          }
          LOG_EVERY_N(INFO, 100000) <<"Prerank PxtrRelativeQuantileScore GetScore exp without f1: "
                    << ", score= " << score_gamora;
          return score_gamora;
        }
        if (enable_quantile_cal_sigmoid_gamora_) {
          for (int i = 0; i < pxtrs_gamora.size(); i++) {
            double center = pxtrs_quan_gamora_[i];
            double relative_quantile = - param_alpha_vec_gamora_[i] * (pxtrs_gamora[i] - center);
            double relative_quantile_adjust = param_sigmoid_b_vec_gamora_[i] /
                    (1 + pow(param_sigmoid_a_vec_gamora_[i], relative_quantile));
            score_gamora *= relative_quantile_adjust;
          }
          LOG_EVERY_N(INFO, 100000) <<"Prerank PxtrRelativeQuantileScore GetScore sigmoid without f1: "
                    << ", score= " << score_gamora;
          return score_gamora;
        }
    }
    LOG_EVERY_N(INFO, 100000) <<"Prerank PxtrRelativeQuantileScore GetScore without f1: "
                        << ", score= " << score_gamora;
    return score_gamora;
  }
};
}  // namespace prerank
}  // namespace slide_leaf
}  // namespace reco
}  // namespace ks
