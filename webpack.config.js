const path = require('path');

module.exports = {
  entry: './src/index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'numpygpt.bundle.js',
    library: 'numpyGPT',
    libraryTarget: 'umd',
    globalObject: 'this'
  },
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      }
    ]
  },
  resolve: {
    extensions: ['.js'],
    fallback: {
      "fs": false,
      "path": require.resolve("path-browserify"),
      "url": require.resolve("url/"),
      "util": require.resolve("util/")
    }
  },
  externals: {
    'fs': 'commonjs fs',
    'path': 'commonjs path'
  },
  devServer: {
    static: {
      directory: path.join(__dirname, 'demo'),
    },
    compress: true,
    port: 8080,
    open: true
  },
  optimization: {
    minimize: true
  }
};