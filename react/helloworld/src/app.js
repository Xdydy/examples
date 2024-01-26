'use strict';

const e = React.createElement;

class MyButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = { clicked: false };
  }

  render() {
    if (this.state.clicked) {
      return 'You clicked this.';
    }

    return e(
      'button',
      { onClick: () => this.setState({ clicked: true }) },
      'Click me'
    );
  }
}

// 渲染组件到 HTML 页面中的 'root' 元素
const domContainer = document.querySelector('#root');
ReactDOM.render(e(MyButton), domContainer);
