---
title: 'Code from Anywhere: Your SSH Guide to Remote Development in PyCharm'
date: '2025-06-26T10:31:30+00:00'
author: gp
layout: post
image: /content/2025/06/pycharm.jpg
categories: [PyCharm, Remote Development]
math: true
mermaid: true
---


If you landed here, you're probably already using a remote (GPU) for your ML tasks, and you're sick of git push from your laptop and the pull on your
powerful GPU machine. Ask me why I know that, and you're probably a PyCharm fan as well 😊.
<br />
If so, you landed in the right place.
<br />
On the web there are several guide how to do remote dev for VS code, but I haven't found one that works for PyCharm, so I decided to write this post.

---

## Introduction

Welcome aboard the world of **SSH-based remote development** with PyCharm! <br /> Imagine editing your DNN locally in PyCharm 
while harnessing the power of a remote Linux server—be it a GPU instance on Runpod, a cluster on Lambda AI, a bare-metal GPU box at Hetzner, or the new [Shadeform.ai](https://www.shadeform.ai?utm_source=genmind.ch) marketplace. 
With a secure SSH tunnel, you get:

* **Massive compute on demand**
* **Consistent, shareable environments**
* **Code safety**—your laptop stays light!

No more fighting environment drift or lugging heavyweight workstations. 
In this guide, we’ll cover every SSH-specific step from host prerequisites to remote interpreter setup, 
plus tips on squeezing GPU performance from budget-friendly servers. Let’s supercharge your Python workflow! ([runpod.io][4], [lambda.ai][5], [hetzner.com][6], [shadeform.ai][7])

---

## Prerequisites

Before you start, make sure your remote host meets PyCharm’s requirements:

* **CPU & RAM**: ≥ 4 vCPUs (x86\_64 or arm64), 8 GB RAM. Higher clock speeds beat more cores for this use case.
* **Disk**: \~10 GB free on local or block storage (avoid NFS/SMB).
* **OS**: Ubuntu 18.04/20.04/22.04, CentOS, Debian, or RHEL.
* **Python & SSH**: A running OpenSSH server on your Linux box and the desired Python version (e.g., `/usr/bin/python3` or a virtualenv).
* **Your public SSH key**: must be deployed on the server. If you are running on Runpod, Lambda, Hetzner, it shall be automatically deployed
* **PyCharm version**: this guide applies to PyCharm 2025.1
---

## Open SSH Configurations

   * On the bottom-left corner of the IDE click on  **Current Interpreter ▸ Add New Interpreter ▸ On SSH** and click **Create SSH configuration**.

![Current Interpreter ▸ Add New Interpreter](/content/2025/06/pycharm1.jpg){: width="300" height="500" }
_Create SSH configuration_

## Fill in Connection Details
   * **Host**: your server’s IP or hostname (in case of the Stanford lab, it would be something like lab-18422c, just copy and paste from your connection string the host name after "@")
   * **Port**: usually `22` (or custom, in case of the Stanford lab, it would be 5084 or anything else provided in the connection string you got after **-p**)
   * **Username**: your SSH user ( (in case of the Stanford lab, it would be something like "scpdxcs", just copy and paste from your connection string the name before "@")
   * click next
<br />
![Connection Details](/content/2025/06/pycharm2.jpg){: width="500" height="400" }
_fill in the connection details_

## Fill in Auth Details
   * **Select**: Key pair and browse for your private SSH key. On a Mac the default location is YOUR_USER/.ssh . It's a hidden folder - Command + Shift + . (period) to show hidden folders on your pop up, if they are not already shown.
   * **Fill the passphrase**: if your SSH private key weas generated with a passphrase. 
   * **Select Save passphrase** 
   * Click **Next**.

If you are connecting using password then select "Password" fill in you password and enable the check box "Save password" if you want to be 
automatically sign in everytime you try to connect again to the remote env.
<br />
![Auth Details](/content/2025/06/pycharm3.jpg){: width="500" height="400" }
_fill in the auth details_

## Introspecting SSH server
   * If you filled all your params correctly you shall see a blank box, just click next
<br />
![Introspecting SSH server](/content/2025/06/pycharm4.jpg){: width="500" height="400" }
_fIntrospecting SSH server_

## Project directory and Python runtime configuration
If you are setting up a remote connection for the Stanford lab, skip to "Using a remote env"
### Using a local env
   * Select the same environment you use locally
   * I leave all the other params as default
   * Click Create
<br />
![Project directory and Python runtime configuration](/content/2025/06/pycharm5.jpg){: width="500" height="400" }
_Project directory and Python runtime configuration - local env_
---
### Using a remote env
   * Conda executable: click the folder brose to select "anaconda/condabin/conda"
   * Select "Create new environment"
   * Environment name: choose your favorite name
   * Python version: 3.12
   * Sync folder: leave it default
   * Check the option "Automatically upload project files"
   * Click Create
<br />
![Project directory and Python runtime configuration](/content/2025/06/sshstanford1.png){: width="500" height="400" }
_Project directory and Python runtime configuration - remote env_
---

## Connecting to the remote environment
  * In PyCharm, on the left, select terminal
  * Click on the dropdown and select the environment you just created (It can take a while before all the file will be transferred)
![Connecting to the remote environment](/content/2025/06/connecttossh.png){: width="500" height="400" }
_Connecting to the remote environment_


You are now remotely connected to your GPU server.
<br />
When you change any file locally, your changes will be automatically uploaded to the server.
This means that you can code locally and run your code on a remote machine with a powerful GPU.
All your ML tasks will be much, much faster!

 
![Remote server terminal](/content/2025/06/connecttossh.png){: width="500" height="400" }
_Remote server terminal_

<br />
If you see a connection timeout error, like the one below, Probably your VM has beed shut down. Go to the portal and reactivate it.

![Connection timeout](/content/2025/06/timeout.png){: width="500" height="400" }
_Connection timeout_

## Running, Testing & Debugging Remotely

* Activate the environment you created during the setup via

```bash
conda activate YOUR_ENV_NAME
```
![Activate your Conda env](/content/2025/06/activatecondaenv.png){: width="500" height="400" }
_Activate your Conda env_

* Update your environment
At this point, if you have a requirements.txt or a yaml file, you'll need to update your Conda environment.
<br />
If you have a yaml file, then: 
```bash
conda env update \
  --name YOUR_ENV_NAME \
  --file YOUR_ENV_YAML \
  --prune
```
In case of the Stanford lab, the name will be "environment_cuda.yml"
<br />
If you have a requirements.txt file, then: 
```bash
pip install -r requirements.txt
```
of course, you need to update your environment only one time. If you log out and then log in again you only need to activate your environment.

> 💡Bonus tip, if you are using a GPU server equipped with an N-Vida GPU you can [install Nvitop](https://genmind.ch/posts/enhance-your-ai-engineer-toolbelt-with-asitop-and-nvitop/) 
> to have an interactive NVIDIA-GPU process viewer, just "pip install nvitop". And to run it "nvitop", you'll have some fun 😊




* **Run Configurations**
  Your existing Run/Debug configs automatically use the SSH interpreter.
* **Breakpoints & Console**
  Set breakpoints locally; the debugger runs over SSH, showing remote stack frames and variables.
* **Remote Python Console**
  Open a Python console that executes commands on the server.
<br />
![Running, Testing & Debugging Remotely](/content/2025/06/pycharm.jpg){: width="500" height="400" }
_Running, Testing & Debugging Remotely_
---

## Deployment & Remote Host Tool Window

* **Auto-Upload on Save**
  * When you make a change to your code, it is automatically deployed to the server; nothing else you have to care about
  * You write code on your laptop, and it gets automatically executed on your GPU server!
  * Optionally, you may want to open a terminal on your GPU server. On the bottom left of the IDE click on **Terminal** and on the drop down choose your newly created SSH connection
<br />
![open a terminal on your GPU server](/content/2025/06/pycharm6.jpg){: width="500" height="400" }
_open a terminal on your GPU server_
---

> ⚠️ Very important:️
> By default, PyCharm’s Deployment plugin only uploads files you modify locally; it does not monitor or pull in remote file changes automatically.
> To download the model, or any other file, you created on the remote GPU machine
> Open the Remote Host window:
> Tools → Deployment → Browse Remote Host
> Locate the file or folder you modified on the server. 
> Right‑click → Download from Here to pull it back into your local project. 

## Licensing & Limitations

* **License**
  SSH interpreters require PyCharm **Professional** (Community Edition doesn’t support them).
* **Limitations**
  Only Linux servers are supported as SSH backends; no remote Windows/macOS interpreters yet, but hey, I hope you are not using a Windows server to test your ML projects!.

---

## Troubleshooting Tips

* **SSH Connection Errors**
  Verify firewall rules, the correct port, and test with `ssh -v user@host`.
* **Interpreter Setup Failures**
  Ensure the SSH config is selected in the SSH Interpreter wizard; see JetBrains support threads for similar issues ([intellij-support.jetbrains.com][13]).
* **Performance Tuning**
  If the remote IDE backend lags, increase its JVM heap in `~/.cache/JetBrains/RemoteDev/*.vmoptions`.

---

## Conclusion & Next Steps

You’ve now unlocked the ability to **code from anywhere**, tapping into remote CPUs or GPUs without leaving PyCharm. 🚀 Next, you might explore:

* **Container-based development** (Docker, Kubernetes)
* **JetBrains Gateway** for zero-install remote work
* **Collaborative coding** via Code With Me


As for compute, cost-effective GPU backends:
- Runpod (pay-per-second from \$0.00011/s) ([runpod.io][4]), 
- Lambda AI (H100 at \$1.85/hr) ([lambda.ai][5]) 
- Hetzner bare-metal GPUs (e.g. RTX-powered servers from €0.295/hr) ([hetzner.com][6])
- The Shadeform.ai marketplace (A100 80 GB PCIe at \$1.20/hr up to H200 SXM5 at \$2.45/hr) ([shadeform.ai][7]). 

Happy AI coding! 😄

[1]: https://www.jetbrains.com/help/pycharm/remote-development-overview.html?utm_source=genmind.ch "Remote development overview | PyCharm Documentation - JetBrains"
[2]: https://www.jetbrains.com/help/pycharm/prerequisites.html?utm_source=genmind.ch "System requirements for remote development | PyCharm - JetBrains"
[3]: https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html?utm_source=genmind.ch "Configure an interpreter using SSH | PyCharm - JetBrains"
[4]: https://www.runpod.io/pricing?utm_source=genmind.ch "Pricing | Runpod GPU cloud computing rates"
[5]: https://lambda.ai/pricing?utm_source=genmind.ch "AI Cloud Pricing | Lambda"
[6]: https://www.hetzner.com/dedicated-rootserver/matrix-gpu/?utm_source=genmind.ch "Server with GPU: for your AI and machine learning projects. - Hetzner"
[7]: https://www.shadeform.ai/?utm_source=genmind.ch "Shadeform - The GPU Cloud Marketplace"
[8]: https://www.jetbrains.com/help/pycharm/create-ssh-configurations.html?utm_source=genmind.ch " | PyCharm Documentation - JetBrains"
[9]: https://www.jetbrains.com/help/pycharm/edit-project-path-mappings-dialog.html?utm_source=genmind.ch "Edit Project Path Mappings dialog | PyCharm - JetBrains"
[10]: https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html?utm_source=genmind.ch "Remote Debugging with PyCharm - JetBrains"
[11]: https://www.jetbrains.com/help/pycharm/tutorial-deployment-in-product.html?utm_source=genmind.ch "Tutorial: Deployment in PyCharm - JetBrains"
[12]: https://www.jetbrains.com/help/pycharm/faq-about-remote-development.html?utm_source=genmind.ch "FAQ about remote development | PyCharm Documentation - JetBrains"
[13]: https://intellij-support.jetbrains.com/hc/en-us/community/posts/10630708200594-Unable-to-add-ssh-interpreter?utm_source=genmind.ch "Unable to add ssh interpreter – IDEs Support (IntelliJ Platform)"
