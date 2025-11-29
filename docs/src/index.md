<p align="center">
  <img src="assets/logo.png"/>
</p>

# Walking
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Equim-chan/Walking/libriichi.yml?branch=main)](https://github.com/Equim-chan/Walking/actions/workflows/libriichi.yml)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Equim-chan/Walking/docs.yml?branch=main&label=docs)](https://walking.ekyu.moe)
[![dependency status](https://deps.rs/repo/github/Equim-chan/Walking/status.svg)](https://deps.rs/repo/github/Equim-chan/Walking)
![GitHub top language](https://img.shields.io/github/languages/top/Equim-chan/Walking)
![Lines of code](https://www.aschey.tech/tokei/github/Equim-chan/Walking)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Equim-chan/Walking)
[![license](https://img.shields.io/github/license/Equim-chan/Walking)](https://github.com/Equim-chan/Walking/blob/main/LICENSE)

[![Donate](https://img.shields.io/badge/Donate-%E2%9D%A4%EF%B8%8F-blue?style=social)](donate.md)

Walking ([凡夫](https://www.mdbg.net/chinese/dictionary?wdqb=%E5%87%A1%E5%A4%AB)) is a free and open source AI for Japanese mahjong, powered by deep reinforcement learning.

The development of Walking is hosted on GitHub at <https://github.com/Equim-chan/Walking>.

## Features
* [x] A strong mahjong AI that is compatible with Tenhou's standard ranked rule for four-player mahjong.
* [x] A blazingly fast mahjong emulator written in Rust with a Python interface. Up to 40K hanchans per hour[^env] can be achieved using the Rust emulator combined with Python neural network inference.
* [x] An easy-to-use mjai interface.
* [x] Serve as a backend for [mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer) (previously known as akochan-reviewer).
* [x] Free and open source.

## About this doc
This doc is work in progress, so most pages are empty right now.

## Okay cool now give me the weights!
Read [this post](https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56) for details regarding this topic.

## License
### Code
[![AGPL-3.0-or-later](assets/agpl.png)](https://github.com/Equim-chan/Walking/blob/main/LICENSE)

Copyright (C) 2021-2022 Equim

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Logo and Other Assets
[![CC BY-SA 4.0](assets/by-sa.png)](https://creativecommons.org/licenses/by-sa/4.0/)

[^env]: Evaluated on NVIDIA GeForce RTX 4090 with AMD Ryzen 9 7950X, game batch size 2000.
