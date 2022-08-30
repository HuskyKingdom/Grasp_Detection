

<!-- PROJECT LOGO -->
<br />
<div align="center" id="readme-top">
  
 <br />

  <h3 align="center">Grasps Detection</h3>

  <p align="center" >
   This assignment includes several attempts to use deep learning techniques to train the neural network to learn a grasp detection, each of the grasp detection value consisting of five elements, they are : ( x ; y ; rotation angle ; opening ; jaw size ). The structure of this report file is divided into the following seven parts:</br>
1. Grasp Detection using CNN + RGB image</br>
2. Evaluations</br>
3. Grasp Detection using CNN + RGB and Depth Image</br>
4. Grasp Detection in the Wild</br>
5. User Instructions</br>
6. References/Related Works</br>
    <br />
    <a href="https://yuhang.topsoftint.com"><strong>View my full bio.</strong></a>
    <br />
    <br />
  </p>
</div>




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#1">Grasp Detection using CNN + RGB image</a></li>
    <li><a href="#2">Evaluations</a></li>
    <li><a href="#3">Grasp Detection using CNN + RGB and Depth Image</a></li>
    <li><a href="#4">Grasp Detection in the Wild</a></li>
    <li><a href="#5">User Instructions</a></li>
    <li><a href="#6">References/Related Works</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Grasp Detection using CNN + RGB image
<p id="1"></p>

Two architectures were used to perform the grasp detection, one of them are being tested and used in the main code for the later sections, the other one, however, due to its worse performance is not used in later sections, but instead its code was provided individually.
Please refer to the user instructions section for more information about the file structure. The following content of this section will explain each of those architectures and illustrate their training loss performances.

1. Direct Grasp Detection with and without batch normalization

This is the version that tested stable and was later used in other sections.
It takes an RGB image tensor of shape (N*3*1024*1024) as input and returns a single grasp detection output array of 5 elements. For each image, it`s ground truth is selected by the one with the highest Jaccard Index value of the current prediction. The architecture of the network is as in below:

<img src="https://raw.githubusercontent.com/HuskyKingdom/Grasp_Detection/main/imgs/1.png">

This architecture is referenced from Joseph Redmon and Anelia Angelova`s paer[0].
This architecture is later improved by us by adding, for each convolutional layers, a batch normalization.
Please see the following figure of its performance of loss function values over episode, note that each episode takes a batch of 5 images as input.

<img src="https://raw.githubusercontent.com/HuskyKingdom/Grasp_Detection/main/imgs/2.png">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

2. Multi-Grasp Detection

This part of the code is *NOT* included in the main code file, instead it was provided individually in an seperate file, please refer to the user instruction section for more details about this.
Please also note that this Multi-Grasp Detection is an implementation of the Multi-Grasp Detection method proposed by Joseph Redmon and Anelia Angelova`s paer[0], because of that, the networks architecture remains the same as Direct Grasp Detection.
The network in this part takes the input tensor of shape (64*3*128*128), where it is dividing the RGB image into a grid of 8*8 cells, each cell has shape (128*128*3), since
  
the original image has shape (1024*1024*3).
For each of the cell, the neural network produces 6 elements output: ( heatmap; x ;
y ; rotation angle ; opening ; jaw size ), where the heatmap is a probability of a single region contains a grasp.

<img src="https://raw.githubusercontent.com/HuskyKingdom/Grasp_Detection/main/imgs/3.png">


### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
