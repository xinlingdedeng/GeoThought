1、使用的是geoqa测试集和gepmetry3k测试集，其中 geoqa测试集754个测试样例，gepmetry3k测试集601个测试样例

2、vllm_server.sh填写model path，加载model,model name为加载成功的模型的命名。

3、vllm_server.sh加载成功后，可以用curl http://localhost:8111/v1/models来验证模型是否加载成功，如果加载成功会出现：

{
  "object": "list",
  "data": [
    {
      "id": "model name",
      "object": "model",
      "created": 1761025634,
      "owned_by": "vllm",
      "root": "model path",
      "parent": null,
      "max_model_len": 65536,
      "permission": [
        {
          "id": "modelperm-08cd7a510e354d7a89f260be795d1dc9",
          "object": "model_permission",
          "created": 1761025634,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}

4、修改run_infer.sh的model_name之后，正确填写prompt_path和image_root的路径，其中geoqa测试集的prompt是geoqa_test_prompts.jsonl，image_root填的是Geo170K的路径。其中gepmetry3k测试集的prompt是geometry3k_test_prompts.jsonl，image_root填的是geometry3k的路径。

5、然后启动文件monitor_inference.sh，monitor_inference.sh文件会循环推理测试集3遍。



