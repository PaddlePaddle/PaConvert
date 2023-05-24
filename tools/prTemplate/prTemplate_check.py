# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sys
import textwrap

import requests

PR_checkTemplate = ["Paddle"]

REPO_TEMPLATE = {
    "Paddle": r"""### PR APIs(.*[^\s].*)### PR Docs(.*[^\s].*)### Description(.*[^\s].*)"""
}


def re_rule(body, CHECK_TEMPLATE):
    PR_RE = re.compile(CHECK_TEMPLATE, re.DOTALL)
    result = PR_RE.search(body)
    return result


def parameter_accuracy(body):
    PR_dic = {}
    PR_APIs = []
    PR_Docs = []
    body = re.sub("\r\n", "", body)
    APIs_end = body.find("### PR Docs")
    Docs_end = body.find("### Description")
    PR_dic["PR APIs"] = body[len("### PR Docs") : APIs_end]
    PR_dic["PR Docs"] = body[APIs_end + 11 : Docs_end]
    message = ""
    for key in PR_dic:
        test_list = PR_APIs if key == "PR APIs" else PR_Docs
        test_list_lower = [l.lower() for l in test_list]
        value = PR_dic[key].strip().split(",")
        if len(value) == 1 and value[0] == "":
            message += f"{key} should be in {test_list}. but now is None."
    return message


def checkComments(url):
    headers = {
        "Authorization": "token " + GITHUB_API_TOKEN,
    }
    response = requests.get(url, headers=headers).json()
    return response


def checkPRTemplate(repo, body, CHECK_TEMPLATE):
    """
    Check if PR's description meet the standard of template
    Args:
        body: PR's Body.
        CHECK_TEMPLATE: check template str.
    Returns:
        res: True or False
    """
    res = False
    note = r"<!-- Demo: https://github.com/PaddlePaddle/PaConvert/pull/71 -->\r\n|<!-- APIs what you’ve done -->|<!-- Describe the docs PR corresponding the APIs -->|<!-- Describe what you’ve done -->"
    if body is None:
        body = ""
    body = re.sub(note, "", body)
    result = re_rule(body, CHECK_TEMPLATE)
    message = ""
    if len(CHECK_TEMPLATE) == 0 and len(body) == 0:
        res = False
    elif result is not None:
        message = parameter_accuracy(body)
        res = True if message == "" else False
    elif result is None:
        res = False
        message = parameter_accuracy(body)
    return res, message


def pull_request_event_template(event, repo, *args, **kwargs):
    pr_effect_repos = PR_checkTemplate
    pr_num = event["number"]
    url = event["comments_url"]
    BODY = event["body"]
    sha = event["head"]["sha"]
    title = event["title"]
    pr_user = event["user"]["login"]
    print(f"receive data : pr_num: {pr_num}, title: {title}, user: {pr_user}")
    if repo in pr_effect_repos:
        CHECK_TEMPLATE = REPO_TEMPLATE[repo]
        global check_pr_template
        global check_pr_template_message
        check_pr_template, check_pr_template_message = checkPRTemplate(
            repo, BODY, CHECK_TEMPLATE
        )

        Template = textwrap.dedent(
            """
                    We list the example in details as follows!!
                    ----------------------------------------------------------------
                    ### PR APIs
                    <!-- APIs what you've done -->
                    torch.transpose
                    torch.Tensor._index_copy
                    torch.permute
                    ...
                    ### PR Docs
                    <!-- Describe the docs PR corresponding the APIs -->
                    https://github.com/PaddlePaddle/docs/pull/_prID
                    ### Description
                    <!-- Describe what you've done -->
                    ...
                    ----------------------------------------------------------------
                    """
        )
        print(f"check_pr_template: {check_pr_template} pr: {pr_num}")
        if check_pr_template is False:
            print("ERROR MESSAGE: Please follow the PR template as follows!")
            print(CHECK_TEMPLATE)
            print(Template)
            sys.exit(7)
        else:
            sys.exit(0)


def get_a_pull(pull_id):
    url = "https://api.github.com/repos/PaddlePaddle/PaConvert/pulls/" + str(pull_id)
    payload = {}
    headers = {
        "Authorization": "token " + GITHUB_API_TOKEN,
        "Accept": "application/vnd.github+json",
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()


def main(org, repo, pull_id):
    pull_info = get_a_pull(pull_id)
    pull_request_event_template(pull_info, repo)


if __name__ == "__main__":
    AGILE_PULL_ID = sys.argv[1]
    GITHUB_API_TOKEN = sys.argv[2]
    main("PaddlePaddle", "Paddle", AGILE_PULL_ID)
