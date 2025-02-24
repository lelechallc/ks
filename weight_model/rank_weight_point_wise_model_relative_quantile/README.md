- 添加请求时间时刻特征
- uplift建模 互动对时长带来的增益
- 修改各项互动权重，突出comment follow

```python
.enrich_attr_by_lua(
    import_item_attr=["interact_label", "click_comment_button",  "action_like_comment", 
                      "action_expand_secondary_comment_count", "action_comment_click_head", 
                      "action_comment_click_nickname", "action_comment_content_copy", 
                      "action_comment_content_forward"],
    function_for_item="calculate",
    export_item_attr=["click_comment_button"],
    lua_script="""
      function calculate()
        local label = 0
        if interact_label > 0 and click_comment_button > 0 then
          label = 1
        end
        if click_comment_button > 0 and (action_like_comment > 0 or action_expand_secondary_comment_count > 0 or action_comment_click_head > 0 or action_comment_click_nickname > 0 or action_comment_content_copy > 0) then
          label = 1
        end
        return label
      end
    """) \

  .enrich_attr_by_lua(
    import_item_attr=["label_names", "label_values", "label_values_bool"],
    function_for_item="extract_cmt_inter",
    export_item_attr=["action_like_comment", "action_expand_secondary_comment_count", "action_comment_click_head", "action_comment_click_nickname", "action_comment_content_copy", "action_comment_content_forward"],
    lua_script="""
      function extract_cmt_inter()
        local names = label_names or {}
        if #names == 0 then
          return 0, 0, 0, 0, 0, 0
        end
        action_like_comment = 0
        action_expand_secondary_comment_count = 0
        action_comment_click_head = 0
        action_comment_click_nickname = 0
        action_comment_content_copy = 0
        action_comment_content_forward = 0
        for i = 1, #label_names do
          if label_names[i] == 335 and label_values_bool[i] > 0 then
            action_like_comment = 1
          end
          if label_names[i] == 336 and label_values[i] > 0 then
            action_expand_secondary_comment_count = 1
          end
          if label_names[i] == 481 and label_values[i] > 0 then
            action_comment_click_head = 1
          end
          if label_names[i] == 482 and label_values[i] > 0 then
            action_comment_click_nickname = 1
          end
          if label_names[i] == 483 and label_values[i] > 0 then
            action_comment_content_copy = 1
          end
          if label_names[i] == 484 and label_values[i] > 0 then
            action_comment_content_forward = 1
          end
        end
        return action_like_comment, action_expand_secondary_comment_count, action_comment_click_head, action_comment_click_nickname, action_comment_content_copy, action_comment_content_forward
      end
    """) \


           dict(name="label_values_bool",
                path="reco_labels.bool_value"),
```